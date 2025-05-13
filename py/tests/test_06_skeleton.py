from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
from neon import Index_3d
from neon.dense import dSpan
from neon.skeleton import Skeleton
import typing


@neon.Container.factory(name='solver')
def get_solver_operator_container(field):
    def setup(loader: neon.Loader):
        loader.set_grid(field.get_grid())

        f_read = loader.get_read_handle(field)

        @wp.func
        def foo(idx: typing.Any):
            wp.neon_print(idx)
            # wp.neon_print(f_read)
            value = wp.neon_read(f_read, idx, 0)
            value = (value +
                     wp.neon_get_x(idx) +
                     wp.neon_get_y(idx) +
                     wp.neon_get_z(idx))
            wp.print(value)

            # value = value + int(idx.x)
            wp.neon_write(f_read, idx, 0, value)

            # print(value)

        loader.declare_kernel(foo)

    return setup


def test_container_int():
    # Get the path of the current script
    script_path = __file__
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    wp.config.mode = "debug"
    wp.config.llvm_cuda = False
    wp.config.verbose = True
    wp.verbose_warnings = True

    wp.init()

    wp.build.set_cpp_standard("c++17")
    wp.build.add_include_directory(script_dir)
    wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

    # It's a good idea to always clear the kernel cache when developing new native or codegen features
    wp.build.clear_kernel_cache()

    # !!! DO THIS BEFORE DEFINING/USING ANY KERNELS WITH CUSTOM TYPES
    neon.init()

    bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                    dev_idx_list=[0])

    dim = Index_3d(1, 1, 3)
    grid = neon.dense.dGrid(bk, dim)
    field = grid.new_field(cardinality=1, dtype=wp.int32)

    def set_value(idx: Index_3d):
        return idx.x + idx.y + idx.z

    for z in range(0, dim.z):
        for y in range(0, dim.y):
            for x in range(0, dim.x):
                idx = Index_3d(x, y, z)
                newValue = set_value(idx)
                field.write(idx=idx,
                            cardinality=0,
                            newValue=newValue)

    field.update_device(0)
    wp.synchronize()

    solver_operator = get_solver_operator_container(field)
    # solver_operator.run(
    #     stream_idx=0,
    #     data_view=neon.DataView.standard(),
    #     container_runtime=neon.Container.ContainerRuntime.neon)
    print('=====================')
    # solver_operator.run(
    #     stream_idx=0,
    #     data_view=neon.DataView.standard(),
    #     container_runtime=neon.Container.ContainerRuntime.neon)

    sk = Skeleton(backend=bk)
    sk.sequence("skeletonTest", [solver_operator])
    sk.run()

    field.update_host(0)
    wp.synchronize()

    for z in range(0, dim.z):
        for y in range(0, dim.y):
            for x in range(0, dim.x):
                idx = Index_3d(x, y, z)
                newValue = set_value(idx)
                newValueRead = field.read(idx=idx,
                                          cardinality=0)
                different = (newValue * 2) - newValueRead
                if different != 0:
                    print(f"Error: {newValue} != {newValueRead}, {different}")

    pass


if __name__ == "__main__":
    test_container_int()
