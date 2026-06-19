import unittest

from env_setup import update_pythonpath

update_pythonpath()

import typing

import warp as wp
import neon

from neon_test_utils import (
    GRID_DIM,
    LEVEL_ONE_PEEL,
    REF_FACTOR,
    SHELL_DEPTH,
    coord_sum,
    export_vti_if_requested,
    init_warp_neon,
    make_peeled_mask,
    make_shell_mask,
    run_container,
    strip_level_zero_under_level_one,
)


@neon.Container.factory(name="MresInitOperator")
def init_operator(field, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f_write = loader.get_mres_write_handle(field)

        @wp.func
        def init_kernel(idx: typing.Any):
            cartesian_idx = wp.neon_global_idx(f_write, idx)
            value = (wp.neon_get_x(cartesian_idx) +
                     wp.neon_get_y(cartesian_idx) +
                     wp.neon_get_z(cartesian_idx))
            wp.neon_write(f_write, idx, 0, value)

        loader.declare_kernel(init_kernel)

    return setup


@neon.Container.factory(name="MresAddOperator")
def add_operator(field, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def add_kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            cartesian_idx = wp.neon_global_idx(f, idx)
            extra = (wp.neon_get_x(cartesian_idx) +
                     wp.neon_get_y(cartesian_idx) +
                     wp.neon_get_z(cartesian_idx))
            wp.neon_write(f, idx, 0, value + extra)

        loader.declare_kernel(add_kernel)

    return setup


@neon.Container.factory(name="MresVerifyOperator")
def verify_operator(field, level, errors):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def verify_kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            cartesian_idx = wp.neon_global_idx(f, idx)
            extra = (wp.neon_get_x(cartesian_idx) +
                     wp.neon_get_y(cartesian_idx) +
                     wp.neon_get_z(cartesian_idx))
            if value != extra * 2:
                wp.atomic_add(errors, 0, 1)

        loader.declare_kernel(verify_kernel)

    return setup


def make_two_level_grid(backend):
    dim = neon.Index_3d(GRID_DIM, GRID_DIM, GRID_DIM)
    block_spacing = REF_FACTOR * REF_FACTOR
    child_span = REF_FACTOR * REF_FACTOR

    level_zero_mask = make_shell_mask(GRID_DIM, SHELL_DEPTH)
    level_one_mask = make_peeled_mask(GRID_DIM // 2, LEVEL_ONE_PEEL)
    level_zero_mask = strip_level_zero_under_level_one(
        level_zero_mask,
        level_one_mask,
        block_spacing,
        child_span,
    )

    grid = neon.mGrid(
        backend,
        dim,
        sparsity_pattern_list=[level_zero_mask, level_one_mask],
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        stencil=[[0, 0, 0], [1, 0, 0]],
    )
    return grid, level_zero_mask


def init_level_zero_host(field, level_zero_mask):
    for z in range(GRID_DIM):
        for y in range(GRID_DIM):
            for x in range(GRID_DIM):
                if level_zero_mask[x, y, z] == 0:
                    continue
                idx = neon.Index_3d(x, y, z)
                field.write(level=0, idx=idx, cardinality=0, newValue=coord_sum(idx))


def verify_level_zero_host(field, level_zero_mask):
    errors = 0
    for z in range(GRID_DIM):
        for y in range(GRID_DIM):
            for x in range(GRID_DIM):
                if level_zero_mask[x, y, z] == 0:
                    continue
                idx = neon.Index_3d(x, y, z)
                expected = coord_sum(idx) * 2
                if expected != field.read(level=0, idx=idx, cardinality=0):
                    errors += 1
    return errors


class TestMresGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon()

    def setUp(self):
        self.backend = neon.Backend(
            runtime=neon.Backend.Runtime.stream,
            dev_idx_list=[0],
        )
        self.grid, self.level_zero_mask = make_two_level_grid(self.backend)
        self.field = self.grid.new_field(
            cardinality=1,
            dtype=wp.int32,
            memory_type=neon.MemoryType.host_device(),
        )
        self.field.zero_run(0, 0)
        self.field.zero_run(1, 0)
        wp.synchronize()

    def test_two_level_add(self):
        init_level_zero_host(self.field, self.level_zero_mask)
        self.field.update_device(0)
        wp.synchronize()

        run_container(init_operator(self.field, level=1))
        run_container(add_operator(self.field, level=1))

        verify_errors = wp.zeros(1, dtype=wp.int32, device="cuda")
        run_container(verify_operator(self.field, level=1, errors=verify_errors))
        self.assertEqual(verify_errors.numpy()[0], 0)

        run_container(add_operator(self.field, level=0))
        self.field.update_host(0)
        wp.synchronize()

        export_vti_if_requested(self.field, "out.vti")
        self.assertEqual(verify_level_zero_host(self.field, self.level_zero_mask), 0)


if __name__ == "__main__":
    unittest.main()
