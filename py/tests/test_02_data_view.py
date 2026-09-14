import unittest

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp

from neon import DataView
from neon_test_utils import init_warp_neon, require_gpu


@require_gpu
class TestDataView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon(verbose=False)

    def test_print_data_views(self):
        @wp.kernel
        def print_kernel(a: DataView, b: DataView, c: DataView):
            wp.NeonDataView_print(a)
            wp.NeonDataView_print(b)
            wp.NeonDataView_print(c)

        with wp.ScopedDevice("cuda:0"):
            views = (
                DataView(DataView.Values.standard),
                DataView(DataView.Values.internal),
                DataView(DataView.Values.boundary),
            )
            wp.launch(print_kernel, dim=1, inputs=list(views))
            wp.synchronize_device()


if __name__ == "__main__":
    unittest.main()
