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

    dim = neon.Index_3d(4, 4, 4)
    maskZero = np.ones((dim.x, dim.y, dim.z), dtype=int)
    # maskZero[3, 3, 0] = 1
    # maskZero[0, 3, 3] = 1
    maskOne = np.ones((2, 2, 2), dtype=int)
    # maskOne[0, 0, 0] = 1
    # maskOne[0, 0, 1] = 0
    # maskOne[0, 1, 0] = 0
    # maskOne[1, 1, 1] = 1

    grid = neon.mGrid(bk, dim,
                      sparsity_pattern_list=[
                          np.ascontiguousarray(maskZero, dtype=np.int32),
                          np.ascontiguousarray(maskOne, dtype=np.int32),
                      ],
                      sparsity_pattern_origins=[neon.Index_3d(0, 0, 0),
                                                neon.Index_3d(0, 0, 0)
                                                ],
                      stencil=[[0, 0, 0], [1, 0, 0]], )
    print(grid)
    field = grid.new_field(cardinality=1, dtype=wp.int32)
    print("Field created")

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
