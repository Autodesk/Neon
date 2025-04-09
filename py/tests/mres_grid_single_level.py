import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
import typing

@neon.Container.factory(name='SolverOperator')
def set_father_to_33(field,level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f_pn = loader.get_mres_write_handle(field)

        @wp.func
        def foo(idx: typing.Any):
            # wp.neon_print(f_read)
            value = wp.neon_read(f_pn, idx, 0)
            cartesianIdx = wp.neon_global_idx(f_pn, idx)
            extra = wp.neon_get_x(cartesianIdx) + wp.neon_get_y(cartesianIdx) + wp.neon_get_z(cartesianIdx)
            #extra = extra * 0 + 1
            if wp.neon_has_children(f_pn, idx):
                wp.printf("Position (%d,%d,%d) read %d extra %d\n",
                          wp.neon_get_x(cartesianIdx),
                          wp.neon_get_y(cartesianIdx),
                          wp.neon_get_z(cartesianIdx), value, extra)
                value = 33
            #wp.print(value)

            # value = value + int(idx.x)
            wp.neon_write(f_pn, idx, 0, value)

            # print(value)

        loader.declare_kernel(foo)

    return setup

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
            value = value + extra
            #wp.print(value)

            # value = value + int(idx.x)
            wp.neon_write(f_read, idx, 0, value)

            # print(value)

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

    dim = neon.Index_3d(16, 16, 16)

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
    # level_one_mask[0, 0, 0] = 1
    # # level_one_mask[1, 0, 0] = 1
    # # level_one_mask[2, 0, 0] = 1
    # # level_one_mask[2, 0, 0] = 1
    # # level_one_mask[m.x-3, 0, 0] = 1
    # # level_one_mask[m.x-2, 0, 0] = 1
    # # level_one_mask[m.x-1, 0, 0] = 1

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
                      sparsity_pattern_list=[ level_zero_mask,level_one_mask ,],
                      sparsity_pattern_origins=[ neon.Index_3d(0, 0, 0),
                                                 neon.Index_3d(0, 0, 0),],
                      stencil=[[0, 0, 0], [1, 0, 0]], )
    print(grid)
    field = grid.new_field(cardinality=1, dtype=wp.int32)
    print("Field created")

    field.zero_run(0,0)
    field.zero_run(1,0)
    wp.synchronize()
    field.update_host(0)

    field.export_vti("topology")

    op_level_zero = set_father_to_33(field, level=0)
    op_level_one = set_father_to_33(field, level=1)
    op_level_zero.run(0)
    op_level_one.run(0)


    wp.synchronize()
    field.update_host(0)
    wp.synchronize()
    field.export_vti("father_mask.vti")

    def set_value(idx: neon.Index_3d):
        return idx.x + idx.y + idx.z

    for z in range(0, dim.z):
        for y in range(0, dim.y):
            for x in range(0, dim.x):
                idx = neon.Index_3d(x, y, z)
                level = 0
                newValue = set_value(idx)
                print(f"@Level {level} Init@({x},{y},{z}): [value] {newValue} ")
                field.write(idx=idx,
                            level=0,
                            cardinality=0,
                            newValue=newValue)

    for z in range(0, dim.z//2):
        for y in range(0, dim.y//2):
            for x in range(0, dim.x//2):
                level = 1
                idx = neon.Index_3d(x*2, y*2, z*2)
                newValue = set_value(idx)
                print(f"@Level {level} Init@({x},{y},{z}): [value] {newValue} ")
                field.write(idx=idx,
                            level=level,
                            cardinality=0,
                            newValue=newValue)

    field.export_vti("in.vti")
    #
    field.update_device(0)
    wp.synchronize()

    solver_operator = get_solver_operator_container(field, level=0)
    solver_operator.run(
        stream_idx=0,
        data_view=neon.DataView.standard(),
        container_runtime=neon.Container.ContainerRuntime.neon)
    solver_operator_l1 = get_solver_operator_container(field, level=1)
    solver_operator_l1.run(
        stream_idx=0,
        data_view=neon.DataView.standard(),
        container_runtime=neon.Container.ContainerRuntime.neon)
    print('=====================')
    # print('=====================')
    # solver_operator.run(
    #     stream_idx=0,
    #     data_view=neon.DataView.standard(),
    #     container_runtime=neon.Container.ContainerRuntime.neon)
    #
    field.update_host(0)
    wp.synchronize()
    field.export_vti("out.vti")

    error_detected = False
    for z in range(0, dim.z):
        for y in range(0, dim.y):
            for x in range(0, dim.x):
                idx = neon.Index_3d(x, y, z)
                level  =0
                newValue = set_value(idx)
                newValue = newValue*2
                newValueRead = field.read(level= 0,
                                          idx=idx,
                                          cardinality=0)
                different = (newValue ) - newValueRead
                if different != 0:
                    print(f"@Level {level}  Error@({x},{y},{z}): [expected]{newValue} != {newValueRead}[read], {different}")
                    error_detected = False
    for z in range(0, dim.z//2):
        for y in range(0, dim.y//2):
            for x in range(0, dim.x//2):
                idx = neon.Index_3d(x*2, y*2, z*2)
                level  =1
                newValue = set_value(idx)
                newValue = newValue*2
                newValueRead = field.read(level= 0,
                                          idx=idx,
                                          cardinality=0)
                different = (newValue ) - newValueRead
                if different != 0:
                    print(f"@Level {level}  Error@({x},{y},{z}): [expected]{newValue} != {newValueRead}[read], {different}")
                    error_detected = False
    pass
    if error_detected:
        print("Test failed")
    else:
        print("Test passed")


if __name__ == "__main__":
    # block until getting an input from keyboard
    pid = os.getpid()
    print(f"Process PID: {pid}")
    print("Press any key to continue...")
    # input()
    block_grid_try()
