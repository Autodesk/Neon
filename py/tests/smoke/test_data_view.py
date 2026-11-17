import unittest

from neon_testing import NeonTestCase, require_gpu

import warp as wp
from neon import DataView


class TestDataViewHost(unittest.TestCase):
    def test_values_roundtrip(self):
        for value in (DataView.Values.standard, DataView.Values.internal, DataView.Values.boundary):
            with self.subTest(value=value):
                self.assertEqual(DataView(value).data_view, value.value)

    def test_standard_factory(self):
        self.assertEqual(DataView.standard().data_view, DataView.Values.standard.value)


@require_gpu
class TestDataViewKernel(NeonTestCase):
    def test_print_in_kernel(self):
        @wp.kernel
        def print_kernel(a: DataView, b: DataView, c: DataView):
            wp.NeonDataView_print(a)
            wp.NeonDataView_print(b)
            wp.NeonDataView_print(c)

        views = [DataView(v) for v in (DataView.Values.standard, DataView.Values.internal, DataView.Values.boundary)]
        wp.launch(print_kernel, dim=1, inputs=views, device="cuda:0")
        wp.synchronize_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
