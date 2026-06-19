import unittest

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon

from neon_test_utils import init_warp_neon, require_gpu


@require_gpu
class TestSpan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

        @wp.func
        def user_foo(idx: neon.dense.dIndex):
            wp.neon_print(idx)

        @wp.kernel
        def neon_kernel_test(span: neon.dense.dSpan):
            is_valid = wp.bool(True)
            my_idx = wp.neon_set(span, is_valid)
            if is_valid:
                user_foo(my_idx)

        cls.neon_kernel_test = neon_kernel_test

    def test_launch_over_span(self):
        with wp.ScopedDevice("cuda:0"):
            backend = neon.Backend(
                runtime=neon.Backend.Runtime.stream,
                dev_idx_list=[0],
            )
            grid = neon.dense.dGrid(backend)
            span = grid.get_span(
                neon.Execution.device(),
                0,
                neon.DataView.standard(),
            )
            self.assertIsNotNone(span)
            wp.launch(self.neon_kernel_test, dim=10, inputs=[span])
            wp.synchronize_device()


if __name__ == "__main__":
    unittest.main()
