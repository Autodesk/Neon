import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
import typing


@neon.Container.factory(name='SolverOperator')
def get_solver_operator_container(field,level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f_read = loader.get_mres_write_handle(field)

        @wp.func
        def foo(idx: typing.Any):
            # wp.neon_print(f_read)
            value = wp.neon_read(f_read, idx, 0)
            cartesianIdx = wp.neon_global_idx(f_read, idx)
            extra = wp.neon_get_x(cartesianIdx) + wp.neon_get_y(cartesianIdx) + wp.neon_get_z(cartesianIdx)
            #extra = extra * 0 + 1
            wp.printf("Position (%d,%d,%d) read %d extra %d\n",
                      wp.neon_get_x(cartesianIdx),
                      wp.neon_get_y(cartesianIdx),
                      wp.neon_get_z(cartesianIdx), value, extra)
            value = level+3
            #wp.print(value)
            wp.neon_write(f_read, idx, 0, value)

            # if not wp.neon.neon_has_children(f_read, idx):
            #     # value = value + int(idx.x)
            #     wp.neon_write(f_read, idx, 0, value)
            # else:
            #     value = value *-1
            #     wp.neon_write(f_read, idx, 0, value)


        loader.declare_kernel(foo)

    return setup


def block_grid_try():
    # Get the path of the current script
    script_path = __file__
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    wp.config.mode = "debug"
    wp.config.llvm_cuda = False
    wp.config.verbose = True
    wp.verbose_warnings = True

    wp.init()
    neon.init()

    bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                      dev_idx_list=[0])
    grid_shape = (64, 64, 64)
    dim = neon.Index_3d(grid_shape[0],
                        grid_shape[1],
                        grid_shape[2])
    level_zero_mask = np.zeros((dim.x, dim.y, dim.z), dtype=int)
    level_zero_mask = np.ascontiguousarray(level_zero_mask, dtype=np.int32)
    # loop over all the elements in level_zero_mask and set to one any that have x=0 or y=0 or z=0
    for i in range(dim.x):
        for j in range(dim.y):
            for k in range(dim.z):
                if i == 0 or j == 0 or k == 0:
                    level_zero_mask[i, j, k] = 1
                if i == dim.x-1 or j == dim.y-1 or k == dim.z-1:
                    level_zero_mask[i, j, k] = 1
                if i == 1 or j == 1 or k == 1:
                    level_zero_mask[i, j, k] = 1
                if i == dim.x-2 or j == dim.y-2 or k == dim.z-2:
                    level_zero_mask[i, j, k] = 1
                if (i == 2 or j == 2 or k == 2):
                    level_zero_mask[i, j, k] = 1
                if i == dim.x-3 or j == dim.y-3 or k == dim.z-3:
                    level_zero_mask[i, j, k] = 1
                if i == 3 or j == 3 or k == 3:
                    level_zero_mask[i, j, k] = 1
                if i == dim.x-4 or j == dim.y-4 or k == dim.z-4:
                    level_zero_mask[i, j, k] = 1


    level_one_mask = np.ones((dim.x//2, dim.y//2, dim.z//2), dtype=int)
    m = neon.Index_3d(dim.x // 2, dim.y // 2, dim.z // 2)
    for i in range(dim.x//2):
        for j in range(dim.y//2):
            for k in range(dim.z//2):
                m = neon.Index_3d(dim.x//2,
                                  dim.y//2,
                                  dim.z//2)
                if i == 0 or j == 0 or k == 0:
                    level_one_mask[i, j, k] = 0
                if i == m.x-1 or j == m.y-1 or k == m.z-1:
                    level_one_mask[i, j, k] = 0
                if i == 1 or j == 1 or k == 1:
                    level_one_mask[i, j, k] = 0
                if (i == m.x-2 or j == m.y-2 or k == m.z-2):
                    level_one_mask[i, j, k] = 0

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
    field = grid.new_field(cardinality=1, dtype=wp.int32)
    print("Field created")
    field.export_vti("export_test","ut")
    field.update_device(0)
    wp.synchronize()
    get_solver_operator_container(field, level=0).run(0)
    get_solver_operator_container(field, level=1).run(0)
    field.update_host(0)
    field.export_vti("export_test_after_kernel","ut")



if __name__ == "__main__":
    # block until getting an input from keyboard
    pid = os.getpid()
    print(f"Process PID: {pid}")
    print("Press any key to continue...")
    # input()
    block_grid_try()
