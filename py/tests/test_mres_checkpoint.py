"""Round-trip tests for mGrid topology + field checkpoint I/O."""

import json
import os
import tempfile
import unittest

import ctypes
import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon
from neon.multires.mGrid import mGrid


class TestMresCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.backend = neon.Backend(runtime=neon.Backend.Runtime.stream, gpus=[0])

    def _make_grid(self):
        dim = neon.Index_3d(32, 32, 32)
        coords0 = np.array([[i, 0, 0] for i in range(0, 16, 2)], dtype=np.int32)
        coords1 = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.int32)
        stencil = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        return mGrid.from_active_voxels(
            self.backend,
            dim,
            [coords0, coords1],
            [neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
            stencil,
        )

    def test_float32_cardinality1_roundtrip(self):
        grid = self._make_grid()
        field = grid.new_field(1, float, neon.MemoryType.HOST_DEVICE)
        for level in range(grid.num_levels):
            span = grid.get_span(level, neon.Execution.HOST)
            # Simple host fill via read/write on a few known coords is heavy; use export path fill via C++ forEach in future.
            # Here we write via update after setting one voxel if inside domain.
            idx = neon.Index_3d(4, 0, 0)
            if grid.is_inside_domain(level, idx):
                field.write(level, idx, ctypes.c_int(0), 100.0 + level)

        with tempfile.TemporaryDirectory() as tmp:
            topo = os.path.join(tmp, "grid.ngrid")
            base = os.path.join(tmp, "rho")
            grid.save_topology(topo)
            field.save(base, name="rho")

            grid2 = mGrid.from_topology(self.backend, topo)
            field2 = grid2.load_field(base)
            self.assertEqual(field2.cardinality, 1)

            with open(base + ".json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["name"], "rho")
            self.assertIn("topology_id", meta)

    def test_topology_id_mismatch_rejected(self):
        grid = self._make_grid()
        field = grid.new_field(1, float, neon.MemoryType.HOST_DEVICE)
        with tempfile.TemporaryDirectory() as tmp:
            topo = os.path.join(tmp, "grid.ngrid")
            base = os.path.join(tmp, "rho")
            grid.save_topology(topo)
            field.save(base, name="rho")

            grid2 = self._make_grid()
            field2 = grid2.new_field(1, float, neon.MemoryType.HOST_DEVICE)
            from neon.multires.mField import FieldError

            with self.assertRaises(FieldError):
                field2.load(base)


if __name__ == "__main__":
    unittest.main()
