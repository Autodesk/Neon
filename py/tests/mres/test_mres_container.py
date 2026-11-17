"""Two-level mGrid containers, verified on device and host at both levels."""

import unittest

from neon_testing import NeonTestCase, export_vti_if_requested, grids, operators, require_gpu

import warp as wp
import neon


@require_gpu
class TestMresContainer(NeonTestCase):
    def setUp(self):
        super().setUp()
        self.grid, self.fine_mask, _ = grids.two_level_mgrid(self.backend())
        self.track(self.grid)
        self.field = self.track(self.grid.new_field(
            cardinality=1, dtype=wp.int32, memory_type=neon.MemoryType.host_device()))
        for level in range(2):
            self.field.zero_run(level, 0)
        wp.synchronize()

    def _device_mismatches(self, level, factor):
        errors = wp.zeros(1, dtype=wp.int32, device="cuda")
        self.run_op(operators.mres_count_coord_sum_mismatch(self.field, level=level, factor=factor, errors=errors))
        return int(errors.numpy()[0])

    def _host_mismatches(self, level, factor):
        bad = []
        for idx in grids.active_voxels(self.grid, level, grids.GRID_DIM):
            if self.field.read(level, idx, 0) != factor * grids.coord_sum(idx):
                bad.append((idx.x, idx.y, idx.z))
        return bad

    def test_both_levels_have_active_voxels(self):
        for level in range(2):
            with self.subTest(level=level):
                self.assertGreater(len(grids.active_voxels(self.grid, level, grids.GRID_DIM)), 0)

    def test_set_then_add_on_device(self):
        for level in range(2):
            with self.subTest(level=level):
                self.run_op(operators.mres_set_coord_sum(self.field, level=level))
                self.run_op(operators.mres_add_coord_sum(self.field, level=level))
                self.assertEqual(self._device_mismatches(level, factor=2), 0)

    def test_host_write_device_add_host_read(self):
        for level in range(2):
            for idx in grids.active_voxels(self.grid, level, grids.GRID_DIM):
                self.field.write(level=level, idx=idx, cardinality=0, newValue=grids.coord_sum(idx))
        self.field.update_device(0)
        wp.synchronize()

        for level in range(2):
            self.run_op(operators.mres_add_coord_sum(self.field, level=level))
        self.field.update_host(0)
        wp.synchronize()
        export_vti_if_requested(self.field, "mres_container")

        for level in range(2):
            with self.subTest(level=level):
                self.assertEqual(self._host_mismatches(level, factor=2), [])

    def test_levels_are_independent(self):
        self.run_op(operators.mres_set_coord_sum(self.field, level=0))
        self.field.update_host(0)
        wp.synchronize()
        for idx in grids.active_voxels(self.grid, 1, grids.GRID_DIM):
            self.assertEqual(self.field.read(1, idx, 0), 0, f"level 1 idx=({idx.x},{idx.y},{idx.z})")


if __name__ == "__main__":
    unittest.main()
