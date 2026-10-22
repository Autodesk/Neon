#!/usr/bin/env python3
"""wp.neon_* alias surface (XLB-compat) backed by Carbon's spanless dense builtins."""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpNeonAliases(unittest.TestCase):
    def test_neon_read_write_ngh_global(self) -> None:
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
        carbon.install_neon_aliases()

        system = carbon.System(gate)
        dense = carbon.Dense(system, 6, 6, 6, halo_radius=1)
        field = dense.new_field(1, dtype=wp.float32)

        # Seed each cell with its global x using the Neon-alias write.
        @wp.kernel
        def seed(part: gate.CarbonDensePartition):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(part, ix, iy, iz)
            g = wp.neon_global_idx(part, idx)
            wp.neon_write(part, idx, 0, wp.float32(wp.neon_get_x(g)))

        part0 = field.get_partition(0, 0)
        span = dense.get_span(0, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        wp.launch(seed, dim=(nx, ny, nz), inputs=[part0], device="cuda:0")
        wp.synchronize()

        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        # For interior cells, the +x neighbour value must equal (this cell's global x + 1).
        @wp.kernel
        def check(part: gate.CarbonDensePartition, out: wp.array(dtype=wp.int32)):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(part, ix, iy, iz)
            g = wp.neon_global_idx(part, idx)
            gx = wp.neon_get_x(g)
            if wp.neon_read(part, idx, 0) != wp.float32(gx):
                out[0] = 1
            ngh = wp.neon_ngh_idx(wp.int8(1), wp.int8(0), wp.int8(0))
            unused = wp.bool(False)
            nbr = wp.neon_read_ngh(part, idx, ngh, 0, wp.float32(-1), unused)
            if gx < wp.int32(5) and nbr != wp.float32(gx + wp.int32(1)):
                out[0] = 1

        wp.launch(check, dim=(nx, ny, nz), inputs=[part0, mismatches], device="cuda:0")
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
