"""mGrid topology queries from kernels: global index, has_child, has_parent, finer neighbours."""

import typing
import unittest

import numpy as np

from neon_testing import NeonTestCase, grids, masks, operators, require_gpu

import warp as wp
import neon


@neon.Container.factory(name="NeonTestMresFinerNgh")
def finer_ngh_flags(field, level):
    """c0 = has_child, c1 = has_finer_ngh in -z."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            has_child = 0
            if wp.neon_has_child(f, idx):
                has_child = 1
            finer = 0
            if wp.neon_has_finer_ngh(f, idx, wp.neon_ngh_idx(wp.int8(0), wp.int8(0), wp.int8(-1))):
                finer = 1
            wp.neon_write(f, idx, 0, has_child)
            wp.neon_write(f, idx, 1, finer)

        loader.declare_kernel(kernel)

    return setup


def _read_flags(field, voxels, level):
    return {(v.x, v.y, v.z): (field.read(level, v, 0), field.read(level, v, 1)) for v in voxels}


@require_gpu
class TestMresTopology(NeonTestCase):
    def _field(self, grid, cardinality, num_levels):
        field = self.track(grid.new_field(cardinality=cardinality, dtype=wp.int32,
                                          memory_type=neon.MemoryType.host_device()))
        for level in range(num_levels):
            field.zero_run(level, 0)
        wp.synchronize()
        return field

    def test_global_idx_is_base_resolution_at_every_level(self):
        # At 64^3 all four levels have active voxels (at 32^3 the coarsest is empty).
        dim, widths = 64, (17, 7, 4)
        level_masks = masks.nested_shell_levels(dim, widths)
        grid = self.track(grids.make_mgrid(self.backend(), dim, level_masks))
        num_levels = len(level_masks)
        field = self._field(grid, cardinality=3, num_levels=num_levels)

        for level in range(num_levels):
            self.run_op(operators.mres_write_global_idx(field, level=level, offset=level))
        field.update_host(0)
        wp.synchronize()

        for level in range(num_levels):
            voxels = grids.active_voxels(grid, level, dim)
            with self.subTest(level=level):
                self.assertGreater(len(voxels), 0)
                bad = [(v.x, v.y, v.z) for v in voxels
                       if [field.read(level, v, c) for c in range(3)] != [v.x + level, v.y + level, v.z + level]]
                self.assertEqual(bad, [])

    def test_has_child_and_has_parent(self):
        grid, _, _ = grids.two_level_mgrid(self.backend())
        self.track(grid)
        field = self._field(grid, cardinality=2, num_levels=2)
        for level in range(2):
            self.run_op(operators.mres_topology_flags(field, level=level))
        field.update_host(0)
        wp.synchronize()

        fine = _read_flags(field, grids.active_voxels(grid, 0, grids.GRID_DIM), 0)
        coarse = _read_flags(field, grids.active_voxels(grid, 1, grids.GRID_DIM), 1)

        self.assertTrue(all(child == 0 for child, _ in fine.values()), "finest level cannot have children")
        self.assertTrue(all(parent == 1 for _, parent in fine.values()), "every fine voxel has a parent")
        self.assertTrue(all(parent == 0 for _, parent in coarse.values()), "coarsest level has no parent")
        self.assertGreater(sum(child for child, _ in coarse.values()), 0, "some coarse cells must be refined")

    def test_has_finer_ngh_matches_neighbour_has_child(self):
        # Fine shell around an unrefined core; the coarse level covers everything.
        n = 14
        fine = masks.outer_shell((n, n, n), 4)
        coarse = np.ascontiguousarray(np.ones((n // 2,) * 3), dtype=np.int32)
        grid = self.track(grids.make_mgrid(self.backend(), n, [fine, coarse]))
        field = self._field(grid, cardinality=2, num_levels=2)
        self.run_op(finer_ngh_flags(field, level=1))
        field.update_host(0)
        wp.synchronize()

        flags = _read_flags(field, grids.active_voxels(grid, 1, n), 1)
        self.assertGreater(sum(child for child, _ in flags.values()), 0)
        unrefined = [(x, y, z) for (x, y, z), (child, _) in flags.items() if child == 0]
        self.assertGreater(len(unrefined), 0, "the core must contain unrefined coarse cells")
        self.assertGreater(sum(flags[c][1] for c in unrefined), 0, "some core cell must border the fine shell in -z")
        for (x, y, z) in unrefined:
            below = flags.get((x, y, z - 2))
            if below is None:
                continue
            with self.subTest(cell=(x, y, z)):
                self.assertEqual(flags[(x, y, z)][1], below[0],
                                 "has_finer_ngh(-z) must equal has_child of the -z neighbour")


if __name__ == "__main__":
    unittest.main()
