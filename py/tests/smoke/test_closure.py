"""Warp kernels capturing Neon values in closures."""

import unittest

from neon_testing import NeonTestCase, require_gpu

import warp as wp
import neon
from neon import DataView, Index_3d
from neon.dense import dSpan


@require_gpu
class TestClosure(NeonTestCase):
    def _launch(self, kernel, out):
        wp.launch(kernel, dim=1, inputs=[out], device="cuda:0")
        return out.numpy().tolist()

    def test_closure_captures_index_by_value(self):
        def make_kernel(value: Index_3d):
            @wp.kernel
            def kernel(out: wp.array(dtype=wp.int32)):
                out[0] = wp.neon_get_x(value)
                out[1] = wp.neon_get_y(value)
                out[2] = wp.neon_get_z(value)

            return kernel

        for value in (Index_3d(-1, -2, -3), Index_3d(17, 42, 99)):
            with self.subTest(value=(value.x, value.y, value.z)):
                out = wp.zeros(3, dtype=wp.int32, device="cuda:0")
                self.assertEqual(self._launch(make_kernel(value), out), [value.x, value.y, value.z])

    def test_two_closures_do_not_share_values(self):
        def make_kernel(value: Index_3d):
            @wp.kernel
            def kernel(out: wp.array(dtype=wp.int32)):
                out[0] = wp.neon_get_x(value)

            return kernel

        a = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        b = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        self.assertEqual(self._launch(make_kernel(Index_3d(5, 0, 0)), a), [5])
        self.assertEqual(self._launch(make_kernel(Index_3d(7, 0, 0)), b), [7])

    def test_closure_captures_neon_types(self):
        backend = self.backend()
        grid = self.track(neon.dense.dGrid(backend))
        field = self.track(grid.new_field(cardinality=1, dtype=wp.int32))
        partition = field.get_partition(neon.Execution.device(), 0, neon.DataView.standard())

        span = dSpan()
        span.dataView = DataView(DataView.Values.internal)
        span.z_ghost_radius = 17
        span.z_boundary_radius = 42
        span.max_z_in_domain = 99
        span.span_dim = Index_3d(2, 4, 6)

        def make_kernel(idx, data_view, span, partition):
            @wp.kernel
            def kernel(out: wp.array(dtype=wp.int32)):
                wp.NeonDataView_print(data_view)
                wp.neon_print(span)
                wp.neon_print_dbg(partition)
                out[0] = wp.neon_get_x(idx) + wp.neon_get_y(idx) + wp.neon_get_z(idx)

            return kernel

        out = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        kernel = make_kernel(Index_3d(3, 2, 1), DataView(DataView.Values.boundary), span, partition)
        self.assertEqual(self._launch(kernel, out), [6])


if __name__ == "__main__":
    unittest.main()
