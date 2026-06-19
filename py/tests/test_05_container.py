import unittest

from env_setup import update_pythonpath

update_pythonpath()

import typing

import warp as wp
import neon

from neon_test_utils import coord_sum, init_warp_neon, run_container


@neon.Container.factory(name="DenseAddOperator")
def add_operator(field):
    def setup(loader: neon.Loader):
        loader.set_grid(field.get_grid())
        f = loader.get_read_handle(field)

        @wp.func
        def add_kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            cartesian_idx = wp.neon_global_idx(f, idx)
            extra = (wp.neon_get_x(cartesian_idx) +
                     wp.neon_get_y(cartesian_idx) +
                     wp.neon_get_z(cartesian_idx))
            wp.neon_write(f, idx, 0, value + extra)

        loader.declare_kernel(add_kernel)

    return setup


class TestDenseContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

    def test_two_pass_add(self):
        backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=[0],
        )
        dim = neon.Index_3d(1, 1, 3)
        grid = neon.dense.dGrid(backend, dim)
        field = grid.new_field(cardinality=1, dtype=wp.int32)

        for z in range(dim.z):
            for y in range(dim.y):
                for x in range(dim.x):
                    idx = neon.Index_3d(x, y, z)
                    field.write(idx=idx, cardinality=0, newValue=coord_sum(idx))

        field.update_device(0)
        wp.synchronize()

        op = add_operator(field)
        run_container(op)
        run_container(op)
        field.update_host(0)
        wp.synchronize()

        for z in range(dim.z):
            for y in range(dim.y):
                for x in range(dim.x):
                    idx = neon.Index_3d(x, y, z)
                    expected = coord_sum(idx) * 3
                    read_value = field.read(idx=idx, cardinality=0)
                    self.assertEqual(expected, read_value)


if __name__ == "__main__":
    unittest.main()
