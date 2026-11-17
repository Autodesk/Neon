"""Dense stencil through ``neon_read_ngh`` against a NumPy reference."""

import itertools
import typing
import unittest

import numpy as np

from neon_testing import NeonTestCase, grids, require_gpu

import warp as wp
import neon

CORNERS = list(itertools.product((-1, 1), repeat=3))


@neon.Container.factory(name="NeonTestJacobi")
def jacobi(f_X, f_Y):
    def setup(loader: neon.Loader):
        loader.set_grid(f_Y.get_grid())
        f_x = loader.get_read_handle(f_X)
        f_y = loader.get_write_handle(f_Y)

        @wp.func
        def kernel(idx: typing.Any):
            acc = wp.float32(0)
            offsets = wp.vec2i(-1, 1)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ngh = wp.neon_ngh_idx(wp.int8(offsets[i]), wp.int8(offsets[j]), wp.int8(offsets[k]))
                        valid = wp.bool(False)
                        acc = acc + wp.neon_read_ngh(f_x, idx, ngh, wp.int32(0), wp.float32(0), valid)
            wp.neon_write(f_y, idx, 0, acc / wp.float32(6))

        loader.declare_kernel(kernel)

    return setup


def _reference(x: np.ndarray) -> np.ndarray:
    """x is indexed [x, y, z]; out-of-domain neighbours read as 0."""
    padded = np.pad(x, 1)
    out = np.zeros_like(x)
    nx, ny, nz = x.shape
    for dx, dy, dz in CORNERS:
        out += padded[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz]
    return out / np.float32(6)


@require_gpu
class TestJacobi(NeonTestCase):
    def test_corner_stencil_matches_numpy(self):
        n = 10
        dim = neon.Index_3d(n, n, n)
        stencil = [[0, 0, 0]] + [list(o) for o in CORNERS]
        grid = self.track(neon.dense.dGrid(self.backend(), dim, stencil=stencil))
        f_x = self.track(grid.new_field(cardinality=1, dtype=wp.float32))
        f_y = self.track(grid.new_field(cardinality=1, dtype=wp.float32))

        x = np.fromfunction(lambda i, j, k: i + j + k, (n, n, n), dtype=np.float32)
        for idx in grids.iter_dim(dim):
            f_x.write(idx=idx, cardinality=0, newValue=float(x[idx.x, idx.y, idx.z]))
        f_x.update_device(0)
        wp.synchronize()

        self.run_op(jacobi(f_X=f_x, f_Y=f_y))
        f_y.update_host(0)
        wp.synchronize()

        expected = _reference(x)
        got = np.zeros_like(x)
        for idx in grids.iter_dim(dim):
            got[idx.x, idx.y, idx.z] = f_y.read(idx=idx, cardinality=0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
