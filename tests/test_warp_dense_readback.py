#!/usr/bin/env python3
"""DenseField.to_numpy / update_host / export_vti host readback (multi-GPU aware).

Seeds every device's slab with a coordinate-encoded pattern via the Neon-alias
write path, then checks that the host gather reassembles the full global domain
exactly (this is the multi-GPU slab-placement contract), and that export_vti
emits a well-formed VTI whose payload round-trips.
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonDenseReadback(unittest.TestCase):
    def test_to_numpy_and_export_vti(self) -> None:
        root = _repo_root()
        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp
            import numpy as np
        except ImportError:
            self.skipTest("warp/numpy not installed")

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        import carbon  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")
        carbon.install_neon_aliases()

        nx, ny, nz = 6, 5, 8
        system = carbon.System(gate)
        dense = carbon.Dense(system, nx, ny, nz, halo_radius=1)
        field = dense.new_field(2, dtype=wp.float32)

        # Seed cell (gx,gy,gz) component c with a distinct value; the encoding fits
        # exactly in float32 (max well below 2**24).
        @wp.kernel
        def seed(part: gate.CarbonDensePartition):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(part, ix, iy, iz)
            g = wp.neon_global_idx(part, idx)
            base = wp.float32(
                wp.neon_get_x(g) + wp.int32(100) * wp.neon_get_y(g) + wp.int32(10000) * wp.neon_get_z(g)
            )
            wp.neon_write(part, idx, 0, base)
            wp.neon_write(part, idx, 1, base + wp.float32(1000000.0))

        n_dev = system.get_num_devices()
        for dev in range(n_dev):
            part = field.get_partition(dev, 0)
            span = dense.get_span(dev, 0)
            dx, dy, dz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
            if dx * dy * dz == 0:
                continue
            wp.launch(seed, dim=(dx, dy, dz), inputs=[part], device=f"cuda:{dev}")
        wp.synchronize()

        arr = field.to_numpy()
        self.assertEqual(arr.shape, (2, nx, ny, nz))

        xs = np.arange(nx).reshape(nx, 1, 1)
        ys = np.arange(ny).reshape(1, ny, 1)
        zs = np.arange(nz).reshape(1, 1, nz)
        expected0 = (xs + 100 * ys + 10000 * zs).astype(np.float32)
        expected = np.stack([expected0, expected0 + 1000000.0], axis=0)
        self.assertTrue(np.array_equal(arr, expected), "multi-GPU host gather mismatch")

        # update_host stores the same mirror; export_vti round-trips the payload.
        field.update_host(0)
        self.assertTrue(np.array_equal(field._host_mirror, expected))

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "field.vti")
            field.export_vti(path, "u")
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn(f'WholeExtent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"', text)
            self.assertIn('Name="u"', text)
            self.assertIn('NumberOfComponents="2"', text)
            # Payload is x-fastest, y, z, components innermost.
            payload = text.split('format="ascii">')[1].split("</DataArray>")[0].split()
            vals = np.array([float(v) for v in payload], dtype=np.float32)
            self.assertEqual(vals.size, 2 * nx * ny * nz)
            expected_flat = np.transpose(expected, (3, 2, 1, 0)).reshape(-1)
            self.assertTrue(np.array_equal(vals, expected_flat))

        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
