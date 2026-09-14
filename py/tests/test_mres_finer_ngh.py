"""Manual demo: finer-neighbor halo behavior. Not run by unittest discover."""

import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
import typing



@neon.Container.factory(name='test')
def setHalos(field,level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f = loader.get_mres_write_handle(field)
        dtype = field.dtype

        @wp.func
        def device(idx: typing.Any):
            # global_point =  wp.neon_global_idx(f, idx)

            are_we_a_halo_cell = wp.neon_has_child(f, idx)
            if are_we_a_halo_cell:
                # HERE: we are a halo cell so we just exit
                wp.neon_write(f, idx, 0, 200)
                return
        loader.declare_kernel(device)

    return kernel

@neon.Container.factory(name='test')
def foo(field,level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f = loader.get_mres_write_handle(field)
        dtype = field.dtype

        @wp.func
        def device(idx: typing.Any):
            # global_point =  wp.neon_global_idx(f, idx)
            wp.print("I AM HERE\n")
            wp.neon_print_log(f, True)
            are_we_a_halo_cell = wp.neon_has_child(f, idx)
            if are_we_a_halo_cell:
                # HERE: we are a halo cell so we just exit
                wp.neon_write(f, idx, 0, -99)
                wp.print(44)
                return

            pull_direction = wp.neon_ngh_idx(wp.int8(0), wp.int8(0), wp.int8(-1))
            if wp.neon_has_finer_ngh(f, idx, pull_direction):
                wp.printf("level %d \n",wp.neon_level(f))
                wp.neon_write(f, idx, 0, -33)
                wp.print(99)

            else:
                wp.print(7777777)
                wp.neon_write(f, idx, 0, 77777)
        loader.declare_kernel(device)

    return kernel

def block_grid_try():
    # Get the path of the current script
    script_path = __file__
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    # wp.config.mode = "debug"
    # wp.config.llvm_cuda = False
    # wp.config.verbose = True
    # wp.verbose_warnings = True

    wp.init()
    neon.init()

    def peel(dim, idx, peel_level, outwards):
        if outwards:
            xIn =  idx.x <= peel_level or idx.x >= dim.x -1 -peel_level
            yIn =  idx.y <= peel_level or idx.y >= dim.y -1 -peel_level
            zIn =  idx.z <= peel_level or idx.z >= dim.z -1 - peel_level
            return xIn or yIn or zIn
        else:
            xIn = idx.x >= peel_level and idx.x <= dim.x - 1 - peel_level
            yIn = idx.y >= peel_level and idx.y <= dim.y - 1 - peel_level
            zIn = idx.z >= peel_level and idx.z <= dim.z - 1 - peel_level
            return xIn and yIn and zIn

    bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                      dev_idx_list=[0])
    grid_shape = (14, 14, 14)
    dim = neon.Index_3d(grid_shape[0],
                        grid_shape[1],
                        grid_shape[2])
    level_zero_mask = np.zeros((dim.x, dim.y, dim.z), dtype=int)
    level_zero_mask = np.ascontiguousarray(level_zero_mask, dtype=np.int32)
    # loop over all the elements in level_zero_mask and set to one any that have x=0 or y=0 or z=0
    for i in range(dim.x):
        for j in range(dim.y):
            for k in range(dim.z):
                idx = neon.Index_3d(i,j,k)
                val = 0
                if peel(dim, idx, 4, True):
                    val = 1
                level_zero_mask[i, j, k] = val

    m = neon.Index_3d(dim.x // 2, dim.y // 2, dim.z // 2)
    level_one_mask = np.ones((m.x, m.y, m.z), dtype=int)
    for i in range(m.x):
        for j in range(m.x):
            for k in range(m.x):
                idx = neon.Index_3d(i,j,k)
                val = 1
                level_one_mask[i, j, k] = val

    level_one_mask = np.ascontiguousarray(level_one_mask, dtype=np.int32)


    grid = neon.mGrid(bk, dim,
                      sparsity_pattern_list=[
                          level_zero_mask,
                          level_one_mask,
                      ],
                      sparsity_pattern_origins=[neon.Index_3d(0, 0, 0),
                                                neon.Index_3d(0, 0, 0)
                                                ],
                      stencil=[[0, 0, 0], [1, 0, 0]], )
    print(grid)
    A = grid.new_field(cardinality=1, dtype=wp.int32, memory_type=neon.MemoryType.host_device())
    # B = grid.new_field(cardinality=1, dtype=wp.int32, memory_type=neon.MemoryType.host_device())

    print("Field created")


    wp.synchronize()
    # test(field, level=0).run(0)
    foo(A, level=1).run(0)
    wp.synchronize()
    A.update_host(stream=0)
    wp.synchronize()

    A.export_vti("mres_finer_ngh","test")


import unittest
from neon_test_utils import require_gpu


@require_gpu
class TestMresFinerNgh(unittest.TestCase):
    def test_run(self):
        block_grid_try()


if __name__ == "__main__":
    unittest.main()
