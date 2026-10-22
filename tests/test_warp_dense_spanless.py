#!/usr/bin/env python3
"""Spanless dense read/write/set/global_idx (Neon-parity: partition embeds its span)."""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpDenseSpanless(unittest.TestCase):
    def test_spanless_seed_and_read(self) -> None:
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
        dense = carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(2, dtype=wp.float32)

        # Seed using the spanless API only: set() and write() take the partition alone.
        @wp.kernel
        def seed(part: gate.CarbonDensePartition):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(part, ix, iy, iz)
            g = wp.carbon.global_idx(part, idx)
            gx = wp.carbon.get_x(g)
            gy = wp.carbon.get_y(g)
            gz = wp.carbon.get_z(g)
            wp.carbon.write(part, idx, 0, wp.float32(gx + gy + gz))
            wp.carbon.write(part, idx, 1, wp.float32(2) * wp.float32(gx + gy + gz))

        part0 = field.get_partition(0, 0)
        span = dense.get_span(0, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        wp.launch(seed, dim=(nx, ny, nz), inputs=[part0], device="cuda:0")
        wp.synchronize()

        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(part: gate.CarbonDensePartition, out: wp.array(dtype=wp.int32)):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(part, ix, iy, iz)
            g = wp.carbon.global_idx(part, idx)
            base = wp.float32(
                wp.carbon.get_x(g) + wp.carbon.get_y(g) + wp.carbon.get_z(g)
            )
            if wp.carbon.read(part, idx, 0) != base:
                out[0] = 1
            if wp.carbon.read(part, idx, 1) != wp.float32(2) * base:
                out[0] = 1

        wp.launch(check, dim=(nx, ny, nz), inputs=[part0, mismatches], device="cuda:0")
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
