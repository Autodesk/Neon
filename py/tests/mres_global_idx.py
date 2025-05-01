import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
import typing



@neon.Container.factory(name='test')
def test(field,level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f = loader.get_mres_write_handle(field)
        dtype = field.dtype

        @wp.func
        def device(idx: typing.Any):
            # wp.neon_print(f_read)
            value = dtype(level+1)*0
            if wp.neon_has_parent(f, idx):
                value = -value

            for c in range(wp.neon_cardinality(f)):
                wp.neon_write(f, idx, c, value)

            global_point =  wp.neon_global_idx(f, idx)
            # if level == 1:
            #     wp.printf("Position (%d,%d,%d) level %d\n",wp.neon_get_x(global_point),
            #               wp.neon_get_y(global_point),
            #               wp.neon_get_z(global_point), level)
            if level ==  1:
                if wp.neon_is_equal(global_point, 4,4,4) or wp.neon_is_equal(global_point, 6,6,6):
                    wp.neon_print(global_point)
                    global_point3 = wp.neon_global_idx(f, idx)
                    wp.neon_cuda_info()
                    wp.printf(
                        "YESSSSSSS (%d,%d,%d) level %d vs %d \n",
                        wp.neon_get_x(global_point3),
                        wp.neon_get_y(global_point3),
                        wp.neon_get_z(global_point3),
                        level,
                        wp.neon_level(f)
                    )
                    wp.neon_print(idx)
                    for c in range(wp.neon_cardinality(f)):
                        wp.neon_write(f, idx, c, 88)
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
                idx = neon.Index_3d(i,j,k)
                val = 0
                if peel(dim, idx, 6, True):
                    val = 1
                level_zero_mask[i, j, k] = val

    m = neon.Index_3d(dim.x // 2, dim.y // 2, dim.z // 2)
    level_one_mask = np.ones((m.x, m.y, m.z), dtype=int)
    for i in range(m.x):
        for j in range(m.x):
            for k in range(m.x):
                idx = neon.Index_3d(i,j,k)
                val = 0
                if peel(dim, idx, dim.x, True) and peel(dim, idx, 3, False):
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
    field = grid.new_field(cardinality=1, dtype=wp.int32)
    c3_field = grid.new_field(cardinality=3, dtype=wp.float32)

    print("Field created")
    field.export_vti("export_test","ut")
    field.update_device(0)
    c3_field.update_device(0)


    wp.synchronize()
    # test(field, level=0).run(0)
    test(field, level=1).run(0)
    wp.synchronize()
    field.update_host(stream=0)
    wp.synchronize()

    field.export_vti("mres_global_idx","test")


if __name__ == "__main__":
    # block until getting an input from keyboard
    pid = os.getpid()
    print(f"Process PID: {pid}")
    print("Press any key to continue...")
    # input()
    block_grid_try()
