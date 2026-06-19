import unittest

from env_setup import update_pythonpath

update_pythonpath()

import typing

import numpy as np
import warp as wp
import neon

from neon_test_utils import (
    FP16_TOLERANCE,
    init_warp_neon_once,
    mres_fp16_available,
    mres_fp16_container_available,
    run_container,
)


def make_two_level_grid(backend, dim_size=8):
    dim = neon.Index_3d(dim_size, dim_size, dim_size)
    level_zero_mask = np.ones((dim.x, dim.y, dim.z), dtype=np.int32)

    level_one_dim = dim.x // 2
    level_one_mask = np.zeros((level_one_dim, level_one_dim, level_one_dim), dtype=np.int32)
    center = level_one_dim // 2 - 1
    level_one_mask[center, center, center] = 1

    origin = center * 4
    level_zero_mask[origin:origin + 4, origin:origin + 4, origin:origin + 4] = 0
    level_zero_mask = np.ascontiguousarray(level_zero_mask, dtype=np.int32)
    level_one_mask = np.ascontiguousarray(level_one_mask, dtype=np.int32)

    return neon.mGrid(
        backend,
        dim,
        sparsity_pattern_list=[level_zero_mask, level_one_mask],
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        stencil=[[0, 0, 0]],
    )


def make_fp16_field(backend=None):
    if backend is None:
        backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=[0],
        )
    field = make_two_level_grid(backend).new_field(
        cardinality=1,
        dtype=wp.float16,
        memory_type=neon.MemoryType.host_device(),
    )
    field.zero_run(0, 0)
    field.zero_run(1, 0)
    wp.synchronize()
    return field


def release_field(field):
    if field is not None:
        field.cleanup()
        wp.synchronize()


def assert_fp16_close(test_case, actual, expected):
    test_case.assertLessEqual(
        abs(float(actual) - float(expected)),
        FP16_TOLERANCE,
        f"expected {expected}, got {actual}",
    )


@neon.Container.factory(name="Fp16InitOperator")
def fp16_init_operator(field, level, fill_value):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f_write = loader.get_mres_write_handle(field)

        @wp.func
        def init_kernel(idx: typing.Any):
            wp.neon_write(f_write, idx, 0, wp.float16(fill_value))

        loader.declare_kernel(init_kernel)

    return setup


@neon.Container.factory(name="Fp16ScaleOperator")
def fp16_scale_operator(field, level, scale):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f_write = loader.get_mres_write_handle(field)

        @wp.func
        def scale_kernel(idx: typing.Any):
            value = wp.neon_read(f_write, idx, 0)
            wp.neon_write(f_write, idx, 0, value * wp.float16(scale))

        loader.declare_kernel(scale_kernel)

    return setup


@neon.Container.factory(name="Fp16VerifyOperator")
def fp16_verify_operator(field, level, expected, errors):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f_read = loader.get_mres_write_handle(field)
        expected_value = wp.float16(expected)
        tolerance = wp.float16(FP16_TOLERANCE)

        @wp.func
        def verify_kernel(idx: typing.Any):
            value = wp.neon_read(f_read, idx, 0)
            if wp.abs(value - expected_value) > tolerance:
                wp.atomic_add(errors, 0, 1)

        loader.declare_kernel(verify_kernel)

    return setup


def verify_level(field, level, expected):
    errors = wp.zeros(1, dtype=wp.int32, device="cuda")
    run_container(fp16_verify_operator(field, level=level, expected=expected, errors=errors))
    return errors.numpy()[0]


@unittest.skipUnless(mres_fp16_available(), "mGrid fp16 bindings are not available")
class TestMresFp16Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon_once()
        cls.backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=[0],
        )

    def setUp(self):
        self.field = make_fp16_field(self.backend)

    def tearDown(self):
        release_field(self.field)
        self.field = None

    def test_level0_fill_and_host_read(self):
        self.field.fill_run(0, np.float16(1.5), 0)
        self.field.update_host(0)
        value = self.field.read(0, neon.Index_3d(0, 0, 0), 0)
        assert_fp16_close(self, value, 1.5)

    def test_level1_fill_and_host_read(self):
        self.field.fill_run(1, np.float16(2.0), 0)
        self.field.update_host(0)
        value = self.field.read(1, neon.Index_3d(4, 4, 4), 0)
        assert_fp16_close(self, value, 2.0)


@unittest.skipUnless(mres_fp16_container_available(), "mGrid fp16 container bindings are not available")
class TestMresFp16Kernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon_once()
        cls.backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=[0],
        )

    def setUp(self):
        release_field(getattr(self, 'field', None))
        self.field = make_fp16_field(self.backend)

    def tearDown(self):
        release_field(self.field)
        self.field = None

    def test_level0_fill_and_scale(self):
        self.field.fill_run(0, np.float16(1.5), 0)
        wp.synchronize()
        run_container(fp16_scale_operator(self.field, level=0, scale=2.0))
        self.assertEqual(verify_level(self.field, level=0, expected=3.0), 0)

    def test_level1_device_fill_and_scale(self):
        self.field.fill_run(1, np.float16(2.0), 0)
        wp.synchronize()
        run_container(fp16_scale_operator(self.field, level=1, scale=2.0))
        self.assertEqual(verify_level(self.field, level=1, expected=4.0), 0)


if __name__ == "__main__":
    unittest.main()
