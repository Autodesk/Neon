"""Round-trip tests for mGrid topology + field checkpoint I/O."""

import json
import os
import tempfile
import unittest

import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon
from neon.multires.mField import FieldError

DIM = 16
STENCIL = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]


def _make_grid(backend, fine_extent):
    """Two-level grid: a fine box [0, fine_extent)^2 x [0, 4) on level 0, everything on level 1."""
    level0 = np.zeros((DIM, DIM, DIM), dtype=np.int32)
    level0[:fine_extent, :fine_extent, :4] = 1
    level1 = np.ones((DIM // 2, DIM // 2, DIM // 2), dtype=np.int32)
    return neon.mGrid(
        backend,
        neon.Index_3d(DIM, DIM, DIM),
        sparsity_pattern_list=[np.ascontiguousarray(level0), np.ascontiguousarray(level1)],
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        stencil=STENCIL,
    )


def _active_voxels(grid, level):
    stride = 1 << level
    voxels = []
    for z in range(0, DIM, stride):
        for y in range(0, DIM, stride):
            for x in range(0, DIM, stride):
                idx = neon.Index_3d(x, y, z)
                if grid.is_inside_domain(level, idx):
                    voxels.append((x, y, z))
    return voxels


def _host_idx(level, x, y, z):
    # bField::getReference divides by the level spacing twice, so per-voxel host
    # read/write at level l needs the base-space coordinate scaled by 2**l.
    s = 1 << level
    return neon.Index_3d(x * s, y * s, z * s)


def _value(level, x, y, z, c):
    return float(x + 10 * y + 100 * z + 1000 * level + 0.5 * c)


class TestMresCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.backend = neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=[0])

    def _fill(self, grid, field, cardinality):
        for level in range(grid.num_levels):
            for (x, y, z) in _active_voxels(grid, level):
                for c in range(cardinality):
                    res = field.write(level, _host_idx(level, x, y, z), c, _value(level, x, y, z, c))
                    self.assertEqual(res, 0)
        field.update_device(0)
        wp.synchronize()

    def _roundtrip(self, cardinality):
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(cardinality, wp.float32, neon.MemoryType.host_device())
        self._fill(grid, field, cardinality)

        with tempfile.TemporaryDirectory() as tmp:
            topo = os.path.join(tmp, "grid.ngrid")
            base = os.path.join(tmp, "rho")
            grid.save_topology(topo)
            field.save(base, name="rho")

            with open(base + ".json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["name"], "rho")
            self.assertEqual(meta["dtype"], "float32")
            self.assertEqual(meta["cardinality"], cardinality)
            self.assertEqual(os.path.getsize(base + ".bin"), meta["num_elements"] * 4)

            grid2 = neon.mGrid.from_topology(self.backend, topo)
            self.assertEqual(grid2.num_levels, grid.num_levels)
            field2 = grid2.load_field(base)
            self.assertEqual(field2.cardinality, cardinality)

            for level in range(grid.num_levels):
                voxels = _active_voxels(grid, level)
                self.assertGreater(len(voxels), 0)
                self.assertEqual(_active_voxels(grid2, level), voxels)
                for (x, y, z) in voxels:
                    for c in range(cardinality):
                        got = field2.read(level, _host_idx(level, x, y, z), c)
                        self.assertEqual(got, _value(level, x, y, z, c),
                                         f"level={level} idx=({x},{y},{z}) c={c}")

    def test_float32_cardinality1_roundtrip(self):
        self._roundtrip(cardinality=1)

    def test_float32_cardinality3_roundtrip(self):
        self._roundtrip(cardinality=3)

    def test_topology_id_mismatch_rejected(self):
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(1, wp.float32, neon.MemoryType.host_device())
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "rho")
            field.save(base, name="rho")

            other = _make_grid(self.backend, fine_extent=12)
            other_field = other.new_field(1, wp.float32, neon.MemoryType.host_device())
            with self.assertRaises(FieldError):
                other_field.load(base)


if __name__ == "__main__":
    unittest.main()
