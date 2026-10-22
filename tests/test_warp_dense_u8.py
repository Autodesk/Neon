#!/usr/bin/env python3
"""uint8 dense fields (LBM mask dtype): fill_run + spanless/neon read-back."""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpDenseU8(unittest.TestCase):
    def test_u8_fill_and_read(self) -> None:
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
        from carbon.warp_types import CarbonDensePartitionU8

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")
        carbon.install_neon_aliases()

        system = carbon.System(gate)
        dense = carbon.Dense(system, 6, 6, 6, halo_radius=1)
        field = dense.new_field(2, dtype=wp.uint8)
        field.fill_run(7)

        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(part: CarbonDensePartitionU8, out: wp.array(dtype=wp.int32)):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(part, ix, iy, iz)
            if wp.carbon.read(part, idx, 0) != wp.uint8(7):
                out[0] = 1
            if wp.neon_read(part, idx, 1) != wp.uint8(7):
                out[0] = 1

        span = dense.get_span(0, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        wp.launch(check, dim=(nx, ny, nz), inputs=[field.get_partition(0, 0), mismatches], device="cuda:0")
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
