#!/usr/bin/env python3
"""Spanless own-level mgrid ops via the fat CarbonMgridPartition (embedded own_span).

Exercises the C.2 surface: set / global_idx / read / write / neighbour on the cross
partition alone (no separately-threaded span), mirroring Neon's single mPartition handle.
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpMgridSpanless(unittest.TestCase):
    def test_spanless_own_level_roundtrip(self) -> None:
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

        system = carbon.System(gate)
        mgrid = carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=2)
        field = mgrid.new_field(1, dtype=wp.float32)

        @wp.func
        def encode(gx: int, gy: int, gz: int):
            return wp.float32(gx + 100 * gy + 10000 * gz)

        # Fill own-level (finest) storage through the fat cross partition, spanless.
        @wp.kernel
        def fill(part: gate.CarbonMgridPartition, span: gate.CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(part, iter_idx, x, y, z)
            g = wp.carbon.global_idx(part, cell)
            wp.carbon.write(
                part,
                cell,
                0,
                encode(wp.carbon.get_x(g), wp.carbon.get_y(g), wp.carbon.get_z(g)),
            )

        @wp.kernel
        def check(
            part: gate.CarbonMgridPartition,
            span: gate.CarbonBlockSpan,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(part, iter_idx, x, y, z)
            g = wp.carbon.global_idx(part, cell)
            gx = wp.carbon.get_x(g)
            gy = wp.carbon.get_y(g)
            gz = wp.carbon.get_z(g)
            # Spanless own read must return what we wrote.
            if wp.carbon.read(part, cell, 0) != encode(gx, gy, gz):
                wp.atomic_add(err_count, 0, 1)
            else:
                wp.atomic_add(ok_count, 0, 1)
            # Spanless own neighbour: when valid, equals the encoded neighbour coordinate.
            nv = wp.carbon.neighbour(part, cell, 1, 0, 0, 0)
            if wp.carbon.valid(nv):
                if wp.carbon.value(nv) != encode(gx + 1, gy, gz):
                    wp.atomic_add(err_count, 0, 1)
                else:
                    wp.atomic_add(ok_count, 0, 1)
            # read_ngh with a sentinel must agree with neighbour().value when valid.
            alt = wp.float32(-1.0)
            got = wp.carbon.read_ngh(part, cell, 1, 0, 0, 0, alt)
            if wp.carbon.valid(nv) and got != wp.carbon.value(nv):
                wp.atomic_add(err_count, 0, 1)

        level = 0
        span = mgrid.get_span(level)
        cross_part = field.get_cross_partition(level)
        dim = (
            int(span.iter_block_count),
            int(span.mem_x),
            int(span.mem_y),
            int(span.mem_z),
        )
        wp.launch(fill, dim=dim, inputs=[cross_part, span], device="cuda:0")
        wp.synchronize()

        err = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        ok = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        wp.launch(check, dim=dim, inputs=[cross_part, span, err, ok], device="cuda:0")
        wp.synchronize()

        err_count = int(err.numpy()[0])
        ok_count = int(ok.numpy()[0])
        self.assertEqual(err_count, 0, f"spanless mgrid mismatches: err={err_count} ok={ok_count}")
        self.assertGreater(ok_count, 0)

        field.delete()
        mgrid.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
