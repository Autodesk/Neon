"""Manual demo: export multi-resolution VTI after parent-operator. Not run by unittest discover."""

import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
import typing


@neon.Container.factory(name='SolverOperator')
def get_solver_operator_container(field, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f_read = loader.get_mres_write_handle(field)
        dtype = field.dtype
        @wp.func
        def foo(idx: typing.Any):
            # wp.neon_print(f_read)
            value = wp.neon_read(f_read, idx, 0)
            cartesianIdx = wp.neon_global_idx(f_read, idx)
            extra = wp.neon_get_x(cartesianIdx) + wp.neon_get_y(cartesianIdx) + wp.neon_get_z(cartesianIdx)

            #extra = extra * 0 + 1
            for c in range(wp.neon_cardinality(f_read)):
                value = dtype(level+3 + c)
                wp.neon_write(f_read, idx, c, dtype(value))
        loader.declare_kernel(foo)

    return setup

@neon.Container.factory(name='has_child_operator')
def has_child_operator(field,level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f = loader.get_mres_write_handle(field)
        dtype = field.dtype
        @wp.func
        def foo(idx: typing.Any):
            # wp.neon_print(f_read)
            value = dtype(level+1)
            if wp.neon_has_child(f, idx):
                value = -value

            for c in range(wp.neon_cardinality(f)):
                wp.neon_write(f, idx, c, value)


        loader.declare_kernel(foo)

    return setup

@neon.Container.factory(name='has_parent_operator')
def has_parent_operator(field,level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f = loader.get_mres_write_handle(field)
        dtype = field.dtype

        @wp.func
        def device(idx: typing.Any):
            # wp.neon_print(f_read)
            value = dtype(level+87)*0
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
                if wp.neon_is_equal(global_point, 2,2,2):
                    wp.neon_print(global_point)
                    global_point3 = wp.neon_global_idx(f, idx)

                    wp.printf(
                        "YESSSSSSS (%d,%d,%d) level %d\n",
                        wp.neon_get_x(global_point3),
                        wp.neon_get_y(global_point3),
                        wp.neon_get_z(global_point3),
                        level,
                    )
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
    field = grid.new_field(cardinality=1, dtype=wp.int32, memory_type=neon.MemoryType.host_device())
    c3_field = grid.new_field(cardinality=3, dtype=wp.float32, memory_type=neon.MemoryType.host_device())

    print("Field created")
    field.export_vti("export_test","ut")
    field.update_device(0)
    c3_field.update_device(0)

   #  wp.synchronize()
   #  get_solver_operator_container(field, level=0).run(0)
   #  get_solver_operator_container(field, level=1).run(0)
   #  get_solver_operator_container(c3_field, level=0).run(0)
   #  get_solver_operator_container(c3_field, level=1).run(0)
   #  field.update_host(0)
   #  c3_field.update_host(0)
   #  wp.synchronize()
   #
   #  field.export_vti("export_test_after_kernel","ut")
   #  c3_field.export_vti("export_test_after_kernel_c3","c3")
   #
   #  # ----------------------
   #
   #  wp.synchronize()
   #  has_child_operator(field, level=0).run(0)
   #  has_child_operator(field, level=1).run(0)
   #  #has_child_operator(c3_field, level=0).run(0)
   #  #has_child_operator(c3_field, level=1).run(0)
   #  wp.synchronize()
   #  field.update_host(stream=0)
   #  c3_field.update_host(stream=0)
   #  wp.synchronize()
   #
   #  field.export_vti("export_test_after_kernel_has_child_mask","ut")
   #
   # # ----------------------

    wp.synchronize()
    # has_parent_operator(field, level=0).run(0)
    has_parent_operator(field, level=1).run(0)
    wp.synchronize()
    field.update_host(stream=0)
    wp.synchronize()

    field.export_vti("export_test_after_has_parent_operator","has_parent_operator")


if __name__ == "__main__":
    block_grid_try()
