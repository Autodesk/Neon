"""The sparse mGrid ingestion paths build the same topology as the dense-mask path."""

import unittest

import numpy as np

from neon_testing import NeonTestCase, grids, require_gpu

import neon

DIM = neon.Index_3d(grids.GRID_DIM, grids.GRID_DIM, grids.GRID_DIM)


def _coords(mask):
    # argwhere yields (x, y, z) rows, matching the mask[x, y, z] convention.
    return np.argwhere(mask == 1).astype(np.int32)


def _inside_sets(grid):
    out = {}
    for level in range(grid.num_levels):
        out[level] = {(v.x, v.y, v.z) for v in grids.active_voxels(grid, level, grids.GRID_DIM)}
    return out


@require_gpu
class TestMresSparse(NeonTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bk = cls.class_backend()
        cls.masks = grids.two_level_masks()
        cls.dense = cls.track_class(grids.make_mgrid(cls.bk, grids.GRID_DIM, cls.masks))
        cls.dense_inside = _inside_sets(cls.dense)

    def test_dense_reference_is_not_trivial(self):
        self.assertFalse(self.dense.is_sparse)
        for level in range(2):
            self.assertGreater(len(self.dense_inside[level]), 0)

    def test_from_active_voxels_matches_dense(self):
        sparse = self.track(neon.mGrid.from_active_voxels(
            self.bk, DIM, [_coords(m) for m in self.masks],
            [grids.ORIGIN] * len(self.masks), grids.DEFAULT_STENCIL))
        self.assertTrue(sparse.is_sparse)
        self.assertEqual(sparse.num_levels, self.dense.num_levels)
        self.assertEqual(_inside_sets(sparse), self.dense_inside)

    def test_builder_matches_dense(self):
        builder = neon.mGridSparseBuilder(self.bk, DIM, num_levels=len(self.masks), stencil=grids.DEFAULT_STENCIL)
        # Level 0 one voxel at a time, level 1 in bulk, to exercise both paths.
        for x, y, z in _coords(self.masks[0]):
            builder.register_voxel(0, int(x), int(y), int(z))
        builder.register_voxels(1, _coords(self.masks[1]))
        sparse = self.track(builder.build())
        self.assertEqual(sparse.num_levels, self.dense.num_levels)
        self.assertEqual(_inside_sets(sparse), self.dense_inside)

    def test_builder_registration_counts(self):
        builder = neon.mGridSparseBuilder(self.bk, DIM, num_levels=2, stencil=grids.DEFAULT_STENCIL)
        builder.register_voxel(0, 0, 0, 0)
        builder.register_voxel(0, 1, 0, 0)
        builder.register_voxels(1, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32))
        self.assertEqual(builder.num_registered(0), 2)
        self.assertEqual(builder.num_registered(1), 2)


if __name__ == "__main__":
    unittest.main()
