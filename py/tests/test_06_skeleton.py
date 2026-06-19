import unittest

from env_setup import update_pythonpath

update_pythonpath()

import typing

import warp as wp
import neon
from neon import Index_3d
from neon.skeleton import Skeleton

from neon_test_utils import coord_sum, gpu_count, init_warp_neon


@neon.Container.factory(name="SkeletonAddOperator")
def add_operator(field):
    def setup(loader: neon.Loader):
        loader.set_grid(field.get_grid())
        f = loader.get_read_handle(field)

        @wp.func
        def add_kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            global_idx = wp.neon_global_idx(f, idx)
            value = (value +
                     wp.neon_get_x(global_idx) +
                     wp.neon_get_y(global_idx) +
                     wp.neon_get_z(global_idx))
            wp.neon_write(f, idx, 0, value)

        loader.declare_kernel(add_kernel)

    return setup


@unittest.skipUnless(gpu_count() >= 1, "CUDA GPU not available")
class TestSkeleton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

    def test_skeleton_sequence_updates_field(self):
        device_count = max(1, min(2, gpu_count()))
        backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=list(range(device_count)),
        )
        dim = Index_3d(10, 10, 6)
        grid = neon.dense.dGrid(backend, dim)
        field = grid.new_field(cardinality=1, dtype=wp.int32)

        for z in range(dim.z):
            for y in range(dim.y):
                for x in range(dim.x):
                    idx = Index_3d(x, y, z)
                    field.write(idx=idx, cardinality=0, newValue=coord_sum(idx))

        field.update_device(0)
        wp.synchronize()

        skeleton = Skeleton(backend=backend)
        skeleton.sequence("skeletonTest", [add_operator(field)])
        skeleton.run()
        field.update_host(0)
        wp.synchronize()

        for z in range(dim.z):
            for y in range(dim.y):
                for x in range(dim.x):
                    idx = Index_3d(x, y, z)
                    expected = coord_sum(idx) * 2
                    read_value = field.read(idx=idx, cardinality=0)
                    self.assertEqual(expected, read_value)


if __name__ == "__main__":
    unittest.main()
