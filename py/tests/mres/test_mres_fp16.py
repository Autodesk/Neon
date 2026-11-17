import typing
import unittest

import numpy as np

from neon_testing import (
    NeonTestCase,
    mres_fp16_available,
    mres_fp16_container_available,
    require_gpu,
)

import warp as wp
import neon

FP16_TOLERANCE = 1e-2
DIM = 8


def _two_level_grid(backend):
    """Fully refined 8^3 except one coarse cell in the middle."""
    fine = np.ones((DIM, DIM, DIM), dtype=np.int32)
    coarse_dim = DIM // 2
    coarse = np.zeros((coarse_dim,) * 3, dtype=np.int32)
    center = coarse_dim // 2 - 1
    coarse[center, center, center] = 1
    origin = center * 4
    fine[origin:origin + 4, origin:origin + 4, origin:origin + 4] = 0
    return neon.mGrid(
        backend,
        neon.Index_3d(DIM, DIM, DIM),
        sparsity_pattern_list=[np.ascontiguousarray(fine), np.ascontiguousarray(coarse)],
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0)] * 2,
        stencil=[[0, 0, 0]],
    )


@neon.Container.factory(name="NeonTestMresFp16Scale")
def fp16_scale(field, level, scale):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            wp.neon_write(f, idx, 0, wp.neon_read(f, idx, 0) * wp.float16(scale))

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresFp16CountMismatch")
def fp16_count_mismatch(field, level, expected, errors):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_read_handle(field)
        expected_value = wp.float16(expected)
        tolerance = wp.float16(FP16_TOLERANCE)

        @wp.func
        def kernel(idx: typing.Any):
            if wp.abs(wp.neon_read(f, idx, 0) - expected_value) > tolerance:
                wp.atomic_add(errors, 0, 1)

        loader.declare_kernel(kernel)

    return setup


class _Fp16Case(NeonTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bk = cls.class_backend()
        cls.grid = cls.track_class(_two_level_grid(cls.bk))

    def setUp(self):
        super().setUp()
        self.field = self.track(self.grid.new_field(cardinality=1, dtype=wp.float16,
                                                    memory_type=neon.MemoryType.host_device()))
        for level in range(2):
            self.field.zero_run(level, 0)
        wp.synchronize()

    def assertFp16Close(self, actual, expected):
        self.assertLessEqual(abs(float(actual) - float(expected)), FP16_TOLERANCE,
                             f"expected {expected}, got {actual}")


@require_gpu
@unittest.skipUnless(mres_fp16_available(), "mGrid fp16 bindings are not available")
class TestMresFp16Host(_Fp16Case):
    def test_fill_and_host_read(self):
        for level, value, probe in ((0, 1.5, neon.Index_3d(0, 0, 0)), (1, 2.0, neon.Index_3d(2, 2, 2))):
            with self.subTest(level=level):
                self.field.fill_run(level, np.float16(value), 0)
                self.field.update_host(0)
                wp.synchronize()
                self.assertFp16Close(self.field.read(level, probe, 0), value)


@require_gpu
@unittest.skipUnless(mres_fp16_container_available(), "mGrid fp16 container bindings are not available")
class TestMresFp16Kernels(_Fp16Case):
    def _mismatches(self, level, expected):
        errors = wp.zeros(1, dtype=wp.int32, device="cuda")
        self.run_op(fp16_count_mismatch(self.field, level=level, expected=expected, errors=errors))
        return int(errors.numpy()[0])

    def test_fill_and_scale(self):
        for level, value in ((0, 1.5), (1, 2.0)):
            with self.subTest(level=level):
                self.field.fill_run(level, np.float16(value), 0)
                wp.synchronize()
                self.run_op(fp16_scale(self.field, level=level, scale=2.0))
                self.assertEqual(self._mismatches(level, 2 * value), 0)


if __name__ == "__main__":
    unittest.main()
