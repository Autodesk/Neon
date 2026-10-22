#!/usr/bin/env python3
"""Spike: can a Warp builtin return a registered type that nests another registered type?

This gates the two-layer binding design. Layer 1 wants `set_and_validate` to return C++'s
``ValidatedIdx { Idx idx; bool valid; }``; layer 2 wants a fat index ``{ Idx idx; Span span; }``
so XLB kernel bodies need no edits. Both hinge on the same unproven codegen capability.

Nesting on the *input* side already ships (``CarbonDensePartition`` embeds a ``CarbonDenseSpan``;
``CarbonMgridPartition`` embeds three), so the four things actually in question are:

  1. registering a ctypes struct whose fields are themselves registered types;
  2. a builtin returning such a struct by value;
  3. handing that value to another builtin, which reads the nested members in C++;
  4. an accessor builtin handing a *nested* registered type back to the kernel.

Correctness is checked against the ordinary span-carrying calls, so a pass means the fat form
addresses identical memory rather than merely compiling.
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpProbeNestedReturn(unittest.TestCase):
    def _bootstrap(self):
        root = _repo_root()
        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp
        except ImportError:
            self.skipTest("warp module not installed")

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        import carbon  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")
        return wp, carbon, gate

    def test_layout_mirrors_cpp(self) -> None:
        """Capability 1, on the host: the nested ctypes structs are laid out as C++ expects.

        The binary constructors in warp_probe.h silently skip their memcpy when the incoming
        byte count does not equal ``sizeof(*this)``, so a padding disagreement would not raise --
        it would hand the kernel a zeroed struct. Checking the sizes here turns that into a
        readable failure instead of a mysterious wrong answer downstream.
        """
        import ctypes

        self._bootstrap()
        from carbon.warp_types import (  # noqa: WPS433
            CarbonDenseIdx,
            CarbonDenseSpan,
            CarbonProbeFatIdx,
            CarbonProbeValidatedIdx,
        )

        self.assertEqual(
            ctypes.sizeof(CarbonProbeFatIdx),
            ctypes.sizeof(CarbonDenseIdx) + ctypes.sizeof(CarbonDenseSpan),
            "fat index should be the plain concatenation of its two members",
        )
        # 3 x int64 then an int32, padded out to 8-byte alignment.
        self.assertEqual(ctypes.sizeof(CarbonProbeValidatedIdx), ctypes.sizeof(CarbonDenseIdx) + 8)
        self.assertEqual(CarbonProbeFatIdx.idx.offset, 0)
        self.assertEqual(CarbonProbeFatIdx.span.offset, ctypes.sizeof(CarbonDenseIdx))

    def test_nested_return_round_trip(self) -> None:
        """Capabilities 2-4, on the device, checked against the span-carrying calls."""
        wp, carbon, gate = self._bootstrap()

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(2, dtype=wp.float32)

        @wp.func
        def encode(ix: int, iy: int, iz: int, c: int):
            return wp.float32((ix * 100 + iy * 10 + iz) * 4 + c)

        @wp.kernel
        def fill_via_fat(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.probe_make_fat(span, ix, iy, iz)
            wp.carbon.probe_fat_write(part, cell, 0, encode(ix, iy, iz, 0))
            wp.carbon.probe_fat_write(part, cell, 1, encode(ix, iy, iz, 1))

        @wp.kernel
        def check(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.probe_make_fat(span, ix, iy, iz)
            thin = wp.carbon.set(span, ix, iy, iz)

            for c in range(2):
                want = encode(ix, iy, iz, c)
                # (2)+(3): the fat value crosses a builtin boundary and is unpacked in C++.
                if wp.carbon.probe_fat_read(part, cell, c) != want:
                    wp.atomic_add(err_count, 0, 1)
                # The fat path must agree with the ordinary span-carrying call.
                elif wp.carbon.read(part, span, thin, c) != want:
                    wp.atomic_add(err_count, 0, 1)
                else:
                    wp.atomic_add(ok_count, 0, 1)

            # (4): pull the nested idx back out and use it in a span-carrying read.
            recovered = wp.carbon.probe_fat_idx(cell)
            if wp.carbon.read(part, span, recovered, 0) != encode(ix, iy, iz, 0):
                wp.atomic_add(err_count, 0, 1)
            else:
                wp.atomic_add(ok_count, 0, 1)

            # ValidatedIdx shape: scalar member via accessor, nested member passed onward.
            v = wp.carbon.probe_set_and_validate(span, ix, iy, iz)
            if not wp.carbon.probe_validated_valid(v):
                wp.atomic_add(err_count, 0, 1)
            elif wp.carbon.read(part, span, wp.carbon.probe_validated_idx(v), 1) != encode(
                ix, iy, iz, 1
            ):
                wp.atomic_add(err_count, 0, 1)
            else:
                wp.atomic_add(ok_count, 0, 1)

        span = dense.get_span()
        part = field.get_partition()
        dim = (int(span.logic_x), int(span.logic_y), int(span.logic_z))
        err = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        ok = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        wp.launch(fill_via_fat, dim=dim, inputs=[part, span], device="cuda:0")
        wp.launch(check, dim=dim, inputs=[part, span, err, ok], device="cuda:0")
        wp.synchronize()

        err_count = int(err.numpy()[0])
        ok_count = int(ok.numpy()[0])
        self.assertEqual(err_count, 0, f"nested-return mismatches: err={err_count} ok={ok_count}")
        self.assertEqual(ok_count, 4 * dim[0] * dim[1] * dim[2])

        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
