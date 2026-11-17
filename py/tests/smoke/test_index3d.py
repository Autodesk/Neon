import unittest

from neon_testing import NeonTestCase, require_gpu

import warp as wp
import neon


@require_gpu
class TestIndex3d(NeonTestCase):
    def test_host_value_reaches_kernel(self):
        @wp.kernel
        def read_components(idx: neon.Index_3d, out: wp.array(dtype=wp.int32)):
            out[0] = wp.neon_get_x(idx)
            out[1] = wp.neon_get_y(idx)
            out[2] = wp.neon_get_z(idx)

        out = wp.zeros(3, dtype=wp.int32, device="cuda:0")
        wp.launch(read_components, dim=1, inputs=[neon.Index_3d(11, 22, 33), out], device="cuda:0")
        self.assertEqual(out.numpy().tolist(), [11, 22, 33])

    def test_device_constructor_and_accessors(self):
        @wp.kernel
        def build(out: wp.array(dtype=wp.int32)):
            idx = wp.neon_idx_3d(17, 42, 99)
            for c in range(3):
                out[c] = wp.neon_get_component(idx, c)
            out[3] = wp.int32(0)
            if wp.neon_is_equal(idx, 17, 42, 99):
                out[3] = wp.int32(1)

        out = wp.zeros(4, dtype=wp.int32, device="cuda:0")
        wp.launch(build, dim=1, inputs=[out], device="cuda:0")
        self.assertEqual(out.numpy().tolist(), [17, 42, 99, 1])

    def test_print_builtins_compile_and_run(self):
        @wp.kernel
        def print_index(idx: neon.Index_3d):
            wp.neon_print(idx)
            wp.neon_print(wp.neon_idx_3d(1, 2, 3))

        wp.launch(print_index, dim=1, inputs=[neon.Index_3d(4, 5, 6)], device="cuda:0")
        wp.synchronize_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
