import unittest

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon

from neon_test_utils import gpu_available, init_warp_neon, require_gpu


@require_gpu
class TestFieldInt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

    def test_read_partition_via_span(self):
        def kernel_generator(field):
            partition = field.get_partition(
                neon.Execution.device(),
                0,
                neon.DataView.standard(),
            )

            @wp.func
            def user_foo(idx: neon.dense.dIndex):
                wp.neon_read(partition, idx, 0)

            @wp.kernel
            def neon_kernel_test(span: neon.dense.dSpan):
                is_valid = wp.bool(True)
                my_idx = wp.neon_set(span, is_valid)
                if is_valid:
                    user_foo(my_idx)

            return neon_kernel_test

        with wp.ScopedDevice("cuda:0"):
            backend = neon.Backend(
                runtime=neon.Backend.Runtime.stream,
                dev_idx_list=[0],
            )
            grid = neon.dense.dGrid(backend, neon.Index_3d(10, 10, 10))
            span = grid.get_span(
                neon.Execution.device(),
                0,
                neon.DataView.standard(),
            )
            field = grid.new_field(cardinality=1, dtype=wp.int32)
            kernel = kernel_generator(field)
            wp.launch(kernel, dim=1, inputs=[span])
            wp.synchronize_device()


if __name__ == "__main__":
    unittest.main()
