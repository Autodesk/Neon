#!/usr/bin/env python3
"""Verify Carbon POD types register and launch under the external-type Warp fork."""

import ctypes
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _require_warp_fork() -> None:
    import warp

    repo = getattr(warp, "__file__", "") or ""
    if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
        raise unittest.SkipTest(
            "Warp external-type fork not installed; run ./scripts/setup_warp.sh"
        )


class CarbonWarpExternalTypes(unittest.TestCase):
    def test_dense_fill_kernel(self) -> None:
        root = _repo_root()
        lib_dir = os.path.join(root, "py", "carbon", "lib", f"linux_{os.uname().machine}")
        lib_path = os.path.join(lib_dir, "libcarbonpy.so")
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp
        except ImportError:
            self.skipTest("warp module not installed")
        _require_warp_fork()

        import carbon  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(1, dtype=wp.float32)

        span = dense.get_span()
        part = field.get_partition()

        @wp.kernel
        def fill(part: gate.CarbonDensePartition, span: gate.CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(ix + iy + iz)
            )

        nx = int(span.iter_x)
        ny = int(span.iter_y)
        nz = int(span.iter_z)
        wp.launch(
            fill,
            dim=(nx, ny, nz),
            inputs=[part, span],
            device="cuda:0",
        )
        wp.synchronize()

        @wp.kernel
        def check(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            got = wp.carbon.read(part, span, cell, 0)
            want = wp.float32(ix + iy + iz)
            if got != want:
                out[0] = 1

        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        wp.launch(
            check,
            dim=(nx, ny, nz),
            inputs=[part, span, mismatches],
            device="cuda:0",
        )
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
