"""Manual demo: father/child mask operators and VTI export. Not run by unittest discover."""

import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import typing

import warp as wp
import neon

from neon_test_utils import (
    GRID_DIM,
    coord_sum,
    export_vti_if_requested,
    init_warp_neon,
    make_peeled_mask,
    make_shell_mask,
    run_container,
)


@neon.Container.factory(name="FatherMaskOperator")
def father_mask_operator(field, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            if wp.neon_has_child(f, idx):
                value = 33
            wp.neon_write(f, idx, 0, value)

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="MresAddOperator")
def add_operator(field, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            cartesian_idx = wp.neon_global_idx(f, idx)
            extra = (wp.neon_get_x(cartesian_idx) +
                     wp.neon_get_y(cartesian_idx) +
                     wp.neon_get_z(cartesian_idx))
            wp.neon_write(f, idx, 0, value + extra)

        loader.declare_kernel(kernel)

    return setup


def main():
    init_warp_neon(verbose=False)

    backend = neon.Backend(
        runtime=neon.Backend.Runtime.stream,
        dev_idx_list=[0],
    )
    dim = neon.Index_3d(GRID_DIM, GRID_DIM, GRID_DIM)
    level_zero_mask = make_shell_mask(GRID_DIM, 4)
    level_one_mask = make_peeled_mask(GRID_DIM // 2, 2)

    grid = neon.mGrid(
        backend,
        dim,
        sparsity_pattern_list=[level_zero_mask, level_one_mask],
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        stencil=[[0, 0, 0], [1, 0, 0]],
    )
    field = grid.new_field(
        cardinality=1,
        dtype=wp.int32,
        memory_type=neon.MemoryType.host_device(),
    )
    field.zero_run(0, 0)
    field.zero_run(1, 0)
    wp.synchronize()
    field.update_host(0)

    export_vti_if_requested(field, "topology")
    run_container(father_mask_operator(field, level=0))
    run_container(father_mask_operator(field, level=1))
    field.update_host(0)
    export_vti_if_requested(field, "father_mask.vti")

    for z in range(GRID_DIM):
        for y in range(GRID_DIM):
            for x in range(GRID_DIM):
                idx = neon.Index_3d(x, y, z)
                field.write(level=0, idx=idx, cardinality=0, newValue=coord_sum(idx))

    coarse = GRID_DIM // 2
    for z in range(coarse):
        for y in range(coarse):
            for x in range(coarse):
                idx = neon.Index_3d(x * 2, y * 2, z * 2)
                field.write(level=1, idx=idx, cardinality=0, newValue=coord_sum(idx))

    field.update_device(0)
    run_container(add_operator(field, level=0))
    run_container(add_operator(field, level=1))
    field.update_host(0)
    export_vti_if_requested(field, "out.vti")


if __name__ == "__main__":
    main()
