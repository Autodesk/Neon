import unittest

from neon_testing import NeonTestCase, gpu_count, grids, operators, require_gpu

import warp as wp
import neon


def _fill_coord_sum(field, dim):
    for idx in grids.iter_dim(dim):
        field.write(idx=idx, cardinality=0, newValue=grids.coord_sum(idx))
    field.update_device(0)
    wp.synchronize()


@require_gpu
class TestDenseContainer(NeonTestCase):
    def _check(self, field, dim, factor):
        field.update_host(0)
        wp.synchronize()
        for idx in grids.iter_dim(dim):
            self.assertEqual(field.read(idx=idx, cardinality=0), factor * grids.coord_sum(idx),
                             f"idx=({idx.x},{idx.y},{idx.z})")

    def test_container_rerun_accumulates(self):
        dim = neon.Index_3d(4, 3, 5)
        grid = self.track(neon.dense.dGrid(self.backend(), dim))
        field = self.track(grid.new_field(cardinality=1, dtype=wp.int32))
        _fill_coord_sum(field, dim)

        op = self.track(operators.add_coord_sum(field))
        for _ in range(2):
            op.run(stream_idx=0, data_view=neon.DataView.standard(),
                   container_runtime=neon.Container.ContainerRuntime.neon)
            wp.synchronize()
        self._check(field, dim, factor=3)

    def test_container_context_manager(self):
        dim = neon.Index_3d(1, 1, 3)
        grid = self.track(neon.dense.dGrid(self.backend(), dim))
        field = self.track(grid.new_field(cardinality=1, dtype=wp.int32))
        _fill_coord_sum(field, dim)

        with operators.add_coord_sum(field) as op:
            op.run(stream_idx=0, data_view=neon.DataView.standard(),
                   container_runtime=neon.Container.ContainerRuntime.neon)
            wp.synchronize()
        self.assertFalse(op.container_handle.value)
        self._check(field, dim, factor=2)


@require_gpu
class TestDenseSkeleton(NeonTestCase):
    def test_skeleton_sequence_updates_field(self):
        devices = range(max(1, min(2, gpu_count())))
        backend = self.backend(devices)
        dim = neon.Index_3d(10, 10, 6)
        grid = self.track(neon.dense.dGrid(backend, dim))
        field = self.track(grid.new_field(cardinality=1, dtype=wp.int32))
        _fill_coord_sum(field, dim)

        op = self.track(operators.add_coord_sum(field))
        skeleton = self.track(neon.Skeleton(backend=backend))
        skeleton.sequence("NeonTestDenseSkeleton", [op])
        skeleton.run()
        skeleton.run()
        field.update_host(0)
        wp.synchronize()
        for idx in grids.iter_dim(dim):
            self.assertEqual(field.read(idx=idx, cardinality=0), 3 * grids.coord_sum(idx))


if __name__ == "__main__":
    unittest.main()
