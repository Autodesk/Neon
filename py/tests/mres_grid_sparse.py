"""Equivalence tests for the sparse mGrid ingestion path.

Builds the same two-level grid via the dense API (``neon.mGrid``) and via the
sparse API (``neon.mGrid.from_active_voxels`` and ``neon.mGridSparseBuilder``),
then asserts the resulting grids are structurally identical.
"""

import unittest

from env_setup import update_pythonpath

update_pythonpath()

import numpy as np
import neon

from neon_test_utils import (
    GRID_DIM,
    LEVEL_ONE_PEEL,
    REF_FACTOR,
    SHELL_DEPTH,
    gpu_available,
    init_warp_neon,
    make_peeled_mask,
    make_shell_mask,
    strip_level_zero_under_level_one,
)

STENCIL = [[0, 0, 0], [1, 0, 0]]


def make_masks():
    block_spacing = REF_FACTOR * REF_FACTOR
    child_span = REF_FACTOR * REF_FACTOR

    level_zero_mask = make_shell_mask(GRID_DIM, SHELL_DEPTH)
    level_one_mask = make_peeled_mask(GRID_DIM // 2, LEVEL_ONE_PEEL)
    level_zero_mask = strip_level_zero_under_level_one(
        level_zero_mask,
        level_one_mask,
        block_spacing,
        child_span,
    )
    return level_zero_mask, level_one_mask


def mask_to_coords(mask):
    # np.argwhere returns indices in (axis0, axis1, axis2) == (x, y, z) order,
    # matching the mask[x, y, z] indexing convention used by the dense path.
    return np.argwhere(mask == 1).astype(np.int32)


def build_dense_grid(backend, masks):
    dim = neon.Index_3d(GRID_DIM, GRID_DIM, GRID_DIM)
    return neon.mGrid(
        backend,
        dim,
        sparsity_pattern_list=list(masks),
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        stencil=STENCIL,
    )


def build_sparse_grid(backend, masks):
    dim = neon.Index_3d(GRID_DIM, GRID_DIM, GRID_DIM)
    coords = [mask_to_coords(m) for m in masks]
    return neon.mGrid.from_active_voxels(
        backend,
        dim,
        coords,
        [neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        STENCIL,
    )


def build_sparse_grid_via_builder(backend, masks):
    dim = neon.Index_3d(GRID_DIM, GRID_DIM, GRID_DIM)
    builder = neon.mGridSparseBuilder(backend, dim, num_levels=len(masks), stencil=STENCIL)
    # Register level 0 one voxel at a time, level 1 in bulk, to exercise both paths.
    for x, y, z in mask_to_coords(masks[0]):
        builder.register_voxel(0, int(x), int(y), int(z))
    builder.register_voxels(1, mask_to_coords(masks[1]))
    return builder.build()


def collect_inside_domain(grid, num_levels):
    """Sample is_inside_domain over the whole base-index domain for each level."""
    result = {}
    for level in range(num_levels):
        active = []
        for x in range(GRID_DIM):
            for y in range(GRID_DIM):
                for z in range(GRID_DIM):
                    idx = neon.Index_3d(x, y, z)
                    if grid.is_inside_domain(level, idx):
                        active.append((x, y, z))
        result[level] = set(active)
    return result


@unittest.skipUnless(gpu_available(), "CUDA GPU not available")
class TestMresGridSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

    def setUp(self):
        self.backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=[0],
        )
        self.masks = make_masks()

    def test_from_active_voxels_matches_dense(self):
        dense = build_dense_grid(self.backend, self.masks)
        sparse = build_sparse_grid(self.backend, self.masks)

        self.assertFalse(dense.is_sparse)
        self.assertTrue(sparse.is_sparse)
        self.assertEqual(dense.num_levels, sparse.num_levels)

        dense_inside = collect_inside_domain(dense, dense.num_levels)
        sparse_inside = collect_inside_domain(sparse, sparse.num_levels)
        self.assertEqual(dense_inside, sparse_inside)

    def test_builder_matches_dense(self):
        dense = build_dense_grid(self.backend, self.masks)
        sparse = build_sparse_grid_via_builder(self.backend, self.masks)

        self.assertEqual(dense.num_levels, sparse.num_levels)
        dense_inside = collect_inside_domain(dense, dense.num_levels)
        sparse_inside = collect_inside_domain(sparse, sparse.num_levels)
        self.assertEqual(dense_inside, sparse_inside)

    def test_builder_registration_counts(self):
        dim = neon.Index_3d(GRID_DIM, GRID_DIM, GRID_DIM)
        builder = neon.mGridSparseBuilder(self.backend, dim, num_levels=2, stencil=STENCIL)
        builder.register_voxel(0, 0, 0, 0)
        builder.register_voxel(0, 1, 0, 0)
        builder.register_voxels(1, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32))
        self.assertEqual(builder.num_registered(0), 2)
        self.assertEqual(builder.num_registered(1), 2)


if __name__ == "__main__":
    unittest.main()
