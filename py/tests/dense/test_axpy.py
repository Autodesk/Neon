"""AXPY through the source-generating ``@neon.container`` / ``@neon.kernel`` API."""

import os
import tempfile
import typing
import unittest

from neon_testing import NeonTestCase, grids, require_gpu

import warp as wp
import neon

ALPHA = 2


@neon.container
def axpy(f_X, f_Y, alpha):
    def loading(loader: neon.Loader):
        loader.set_grid(f_Y.get_grid())
        f_x = loader.get_read_handle(f_X)
        f_y = loader.get_write_handle(f_Y)

        @neon.kernel(loader)
        def body(idx: typing.Any):
            for c in range(wp.neon_cardinality(f_x)):
                x = wp.neon_read(f_x, idx, c)
                y = wp.neon_read(f_y, idx, c)
                wp.neon_write(f_y, idx, c, x + alpha * y)

    return loading


def _inputs(idx, c):
    s = grids.coord_sum(idx) + c
    return s, s + 1


@require_gpu
class TestAxpy(NeonTestCase):
    def _run_axpy(self, dim, cardinality, dtype):
        backend = self.backend()
        grid = self.track(neon.dense.dGrid(backend, dim, sparsity=None, stencil=[[0, 0, 0]]))
        f_x = self.track(grid.new_field(cardinality=cardinality, dtype=dtype))
        f_y = self.track(grid.new_field(cardinality=cardinality, dtype=dtype))
        for idx in grids.iter_dim(dim):
            for c in range(cardinality):
                x, y = _inputs(idx, c)
                f_x.write(idx=idx, cardinality=c, newValue=dtype(x))
                f_y.write(idx=idx, cardinality=c, newValue=dtype(y))
        f_x.update_device(0)
        f_y.update_device(0)
        wp.synchronize()

        self.run_op(axpy(f_X=f_x, f_Y=f_y, alpha=dtype(ALPHA)))
        f_y.update_host(0)
        wp.synchronize()

        for idx in grids.iter_dim(dim):
            for c in range(cardinality):
                x, y = _inputs(idx, c)
                self.assertEqual(f_y.read(idx=idx, cardinality=c), x + ALPHA * y,
                                 f"idx=({idx.x},{idx.y},{idx.z}) c={c}")
        return backend, f_x, f_y

    def test_int32_column(self):
        self._run_axpy(neon.Index_3d(1, 1, 100), cardinality=1, dtype=wp.int32)

    def test_float32_cube_multi_component(self):
        self._run_axpy(neon.Index_3d(6, 5, 4), cardinality=3, dtype=wp.float32)

    def test_skeleton_dot_export(self):
        backend, f_x, f_y = self._run_axpy(neon.Index_3d(2, 2, 2), cardinality=1, dtype=wp.int32)
        op = self.track(axpy(f_X=f_x, f_Y=f_y, alpha=wp.int32(ALPHA)))
        skeleton = self.track(neon.Skeleton(backend))
        skeleton.sequence("NeonTestAxpy", [op])
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "axpy")
            skeleton.ioToDot(base, "axpy")
            written = [f for f in os.listdir(tmp) if f.startswith("axpy")]
            self.assertTrue(written, "ioToDot wrote no file")
            self.assertGreater(os.path.getsize(os.path.join(tmp, written[0])), 0)


if __name__ == "__main__":
    unittest.main()
