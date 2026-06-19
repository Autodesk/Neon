import unittest

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon

from neon_test_utils import gpu_available, init_warp_neon, require_gpu


@require_gpu
class TestIndex3d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

    def test_index_print_and_create(self):
        @wp.kernel
        def index_print_kernel(idx: neon.Index_3d):
            wp.neon_print(idx)

        @wp.kernel
        def index_create_kernel():
            idx = wp.neon_idx_3d(17, 42, 99)
            wp.neon_print(idx)

        with wp.ScopedDevice("cuda:0"):
            idx = neon.Index_3d(11, 22, 33)
            wp.launch(index_print_kernel, dim=1, inputs=[idx])
            wp.launch(index_create_kernel, dim=1, inputs=[])
            wp.synchronize_device()


if __name__ == "__main__":
    unittest.main()
