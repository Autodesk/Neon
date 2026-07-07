"""Manual demo: write per-level global indices and export VTI. Not run by unittest discover."""

import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import typing

import warp as wp
import neon

from neon_test_utils import export_vti_if_requested, init_warp_neon, run_container


@neon.Container.factory(name="GlobalIdxOperator")
def global_idx_operator(field, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def device(cell: typing.Any):
            cartesian_idx = wp.neon_global_idx(f, cell)
            for c in range(wp.neon_cardinality(f)):
                val = wp.neon_get_component(cartesian_idx, c) + level
                wp.neon_write(f, cell, c, val)

        loader.declare_kernel(device)

    return setup


def get_peeled_mask(dim, level, width):
    def peel(grid_dim, idx, peel_level, outwards):
        if outwards:
            return (idx.x <= peel_level or idx.x >= grid_dim.x - 1 - peel_level or
                    idx.y <= peel_level or idx.y >= grid_dim.y - 1 - peel_level or
                    idx.z <= peel_level or idx.z >= grid_dim.z - 1 - peel_level)
        return (peel_level <= idx.x <= grid_dim.x - 1 - peel_level and
                peel_level <= idx.y <= grid_dim.y - 1 - peel_level and
                peel_level <= idx.z <= grid_dim.z - 1 - peel_level)

    divider = 2 ** level
    grid_dim = dim if level == 0 else neon.Index_3d(
        dim.x // divider,
        dim.y // divider,
        dim.z // divider,
    )
    mask = np.zeros((grid_dim.x, grid_dim.y, grid_dim.z), dtype=np.int32)
    peel_level = grid_dim.x / width
    for i in range(grid_dim.x):
        for j in range(grid_dim.y):
            for k in range(grid_dim.z):
                idx = neon.Index_3d(i, j, k)
                if peel(grid_dim, idx, peel_level, True):
                    mask[i, j, k] = 1
    return np.ascontiguousarray(mask, dtype=np.int32)


def main():
    init_warp_neon(verbose=False)

    dim = neon.Index_3d(64, 64, 64)
    num_levels = 4
    divider = 2 ** (num_levels - 1)
    coarse_dim = neon.Index_3d(
        dim.x // divider + 1,
        dim.y // divider + 1,
        dim.z // divider + 1,
    )
    levels = [
        get_peeled_mask(dim, 0, 17),
        get_peeled_mask(dim, 1, 7),
        get_peeled_mask(dim, 2, 4),
        np.ascontiguousarray(np.ones((coarse_dim.x, coarse_dim.y, coarse_dim.z), dtype=np.int32)),
    ]

    backend = neon.Backend(
        runtime=neon.Backend.Runtime.stream,
        dev_idx_list=[0],
    )
    grid = neon.mGrid(
        backend,
        dim,
        sparsity_pattern_list=levels,
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0)] * len(levels),
        stencil=[[0, 0, 0], [1, 0, 0]],
    )
    field = grid.new_field(
        cardinality=3,
        dtype=wp.int32,
        memory_type=neon.MemoryType.host_device(),
    )

    run_container(global_idx_operator(field, level=0))
    run_container(global_idx_operator(field, level=1))
    field.update_host(0)
    export_vti_if_requested(field, "mres_global_idx", field_name="test")


import unittest
from neon_test_utils import require_gpu


@require_gpu
class TestMresGlobalIdx(unittest.TestCase):
    def test_run(self):
        main()


if __name__ == "__main__":
    unittest.main()
