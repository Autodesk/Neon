import unittest

from neon_testing import NeonTestCase, grids, operators, require_gpu

import warp as wp
import neon


@require_gpu
class TestBlockContainer(NeonTestCase):
    def _two_pass_add(self, dim):
        grid = self.track(neon.block.bGrid(self.backend(), dim))
        field = self.track(grid.new_field(cardinality=1, dtype=wp.int32))
        for idx in grids.iter_dim(dim):
            field.write(idx=idx, cardinality=0, newValue=grids.coord_sum(idx))
        field.update_device(0)
        wp.synchronize()

        op = operators.add_coord_sum(field)
        self.run_op(op)
        self.run_op(op)
        field.update_host(0)
        wp.synchronize()

        for idx in grids.iter_dim(dim):
            self.assertEqual(field.read(idx=idx, cardinality=0), 3 * grids.coord_sum(idx),
                             f"idx=({idx.x},{idx.y},{idx.z})")

    def test_column(self):
        self._two_pass_add(neon.Index_3d(1, 1, 5))

    def test_multi_block_box(self):
        # Spans more than one 4^3 block along every axis.
        self._two_pass_add(neon.Index_3d(6, 5, 9))


if __name__ == "__main__":
    unittest.main()
