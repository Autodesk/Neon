"""Raw ``wp.launch`` over a dense span and partition (no Container).

A 1-D launch maps thread ids onto the x axis only, so the grids here are
``(n, 1, 1)``; Containers handle 3-D launch configuration themselves.
"""

import unittest

from neon_testing import NeonTestCase, require_gpu

import warp as wp
import neon

N = 8


def _encode(x, y, z):
    return x + 10 * y + 100 * z + 1


@require_gpu
class TestDenseSpan(NeonTestCase):
    def setUp(self):
        super().setUp()
        backend = self.backend()
        self.dim = neon.Index_3d(N, 1, 1)
        self.grid = self.track(neon.dense.dGrid(backend, self.dim))
        self.span = self.grid.get_span(neon.Execution.device(), 0, neon.DataView.standard())

    def test_span_is_valid_for_every_cell(self):
        @wp.kernel
        def count_valid(span: neon.dense.dSpan, count: wp.array(dtype=wp.int32)):
            ok = wp.bool(True)
            wp.neon_set(span, ok)
            if ok:
                wp.atomic_add(count, 0, 1)

        count = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        wp.launch(count_valid, dim=N, inputs=[self.span, count], device="cuda:0")
        self.assertEqual(int(count.numpy()[0]), N)

    def test_partition_write_lands_at_global_idx(self):
        field = self.track(self.grid.new_field(cardinality=1, dtype=wp.int32))
        partition = field.get_partition(neon.Execution.device(), 0, neon.DataView.standard())

        def make_kernel(part):
            @wp.kernel
            def write_global(span: neon.dense.dSpan):
                ok = wp.bool(True)
                idx = wp.neon_set(span, ok)
                if ok:
                    g = wp.neon_global_idx(part, idx)
                    wp.neon_write(part, idx, 0,
                                  wp.neon_get_x(g) + 10 * wp.neon_get_y(g) + 100 * wp.neon_get_z(g) + 1)

            return write_global

        wp.launch(make_kernel(partition), dim=N, inputs=[self.span], device="cuda:0")
        wp.synchronize()
        field.update_host(0)
        wp.synchronize()
        for x in range(N):
            with self.subTest(x=x):
                self.assertEqual(field.read(idx=neon.Index_3d(x, 0, 0), cardinality=0), _encode(x, 0, 0))

    def test_partition_read_in_kernel(self):
        field = self.track(self.grid.new_field(cardinality=1, dtype=wp.int32))
        for x in range(N):
            field.write(idx=neon.Index_3d(x, 0, 0), cardinality=0, newValue=3 * x)
        field.update_device(0)
        wp.synchronize()
        partition = field.get_partition(neon.Execution.device(), 0, neon.DataView.standard())

        def make_kernel(part):
            @wp.kernel
            def sum_values(span: neon.dense.dSpan, total: wp.array(dtype=wp.int32)):
                ok = wp.bool(True)
                idx = wp.neon_set(span, ok)
                if ok:
                    wp.atomic_add(total, 0, wp.neon_read(part, idx, 0))

            return sum_values

        total = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        wp.launch(make_kernel(partition), dim=N, inputs=[self.span, total], device="cuda:0")
        self.assertEqual(int(total.numpy()[0]), 3 * sum(range(N)))


if __name__ == "__main__":
    unittest.main()
