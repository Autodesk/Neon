from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon 
import typing
from typing import Any

# wp.config.print_launches = True

@wp.kernel
def warp_AXPY(
        x: wp.array4d(dtype=Any),
        y: wp.array4d(dtype=Any),
        alpha: Any):
    i, j, k = wp.tid()
    for c in range(x.shape[0]):
        y[c, k, j, i] = x[c, k, j, i] + alpha * y[c, k, j, i]


@neon.Container.factory_v2(name = 'AXPY')
def get_AXPY(f_X, f_Y, alpha: Any):
    def axpy(loader: neon.Loader):
        loader.set_grid(f_Y.get_grid())
        beta = wp.float32(33)
        f_x = loader.get_read_handle(f_X)
        f_y = loader.get_write_handle(f_Y)

        @wp.func
        def foo(idx: typing.Any):
            #            wp.neon_print(idx)
            # wp.neon_print(f_read)
            #for c in range(wp.neon_cardinality(f_x)):
            c = 0
            x = wp.neon_read(f_x, idx, c)
            y = wp.neon_read(f_y, idx, c)
            axpy_res = x + (alpha+beta) * y
            # wp.print(alpha)
            wp.neon_write(f_y, idx, c, axpy_res)
        loader.declare_kernel(foo)

    return axpy
# def get_AXPY(f_X, f_Y, alpha: Any):
#     def axpy(loader: neon.Loader):
#
#         @wp.kernel
#         def kernel(
#                 span: f_X.get_span_type(),
#                 f_x: f_X.get_partitinon_type(),
#                 f_y: f_Y.get_partitinon_type()):
#             is_active = wp.bool(False)
#             myIdx = wp.neon_set(span, is_active)
#
#             @wp.func
#             def foo(idx: typing.Any):
#                 #            wp.neon_print(idx)
#                 # wp.neon_print(f_read)
#                 # for c in range(wp.neon_cardinality(f_x)):
#                 c = 0
#                 x = wp.neon_read(f_x, idx, c)
#                 y = wp.neon_read(f_y, idx, c)
#                 axpy_res = x + alpha * y
#                 # wp.print(alpha)
#                 wp.neon_write(f_y, idx, c, axpy_res)
#
#             if is_active:
#                 # print("NEON-RUNTIME kernel - myIdx: ")
#                 # wp.neon_print(myIdx)
#                 foo(myIdx)
#
#         loader.declare_kernel_v2(kernel)
#
#     return axpy


def execution(nun_devs: int,
              num_card: int,
              dim: neon.Index_3d,
              dtype,
              container_runtime: neon.Container.ContainerRuntime):
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

    dev_idx_list = list(range(nun_devs))
    bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                    dev_idx_list=dev_idx_list)

    grid = neon.dense.dGrid(bk, dim, sparsity=None, stencil=[[0, 0, 0]])
    field_X = grid.new_field(cardinality=num_card, dtype=dtype)
    field_Y = grid.new_field(cardinality=num_card, dtype=dtype)
    field_result = grid.new_field(cardinality=num_card, dtype=dtype)

    # Initialize two 4D arrays on the CPU
    cpu_warp_X = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cpu")
    cpu_warp_Y = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cpu")
    gpu_warp_X = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cuda")
    gpu_warp_Y = wp.empty(shape=(num_card, dim.z, dim.y, dim.x), dtype=dtype, device="cuda")

    alpha = dtype(2)

    def golden_axpy(index: neon.Index_3d, cardinality: int):
        def golden_axpy_input(idx: neon.Index_3d, cardinality: int):
            return (dtype(idx.x + idx.y + idx.z + cardinality), dtype(idx.x + idx.y + idx.z + cardinality + 1))

        x, y = golden_axpy_input(index, cardinality)
        return x, y, x + dtype(alpha) * y

    np_array_X = cpu_warp_X.numpy()
    np_array_Y = cpu_warp_Y.numpy()

    for zi in range(0, dim.z):
        for yi in range(0, dim.y):
            for xi in range(0, dim.x):
                for c in range(0, num_card):
                    idx = neon.Index_3d(xi, yi, zi)
                    # print(f"idx {idx}")
                    x, y, result = golden_axpy(idx, c)
                    field_X.write(idx=idx,
                                  cardinality=c,
                                  newValue=dtype(x))
                    field_Y.write(idx=idx,
                                  cardinality=c,
                                  newValue=dtype(y))
                    field_result.write(idx,
                                       cardinality=c,
                                       newValue=dtype(result))

                    np_array_X[c, zi, yi, xi] = x
                    np_array_Y[c, zi, yi, xi] = y

    field_X.update_device(0)
    field_Y.update_device(0)
    field_result.update_device(0)

    cpu_warp_X = wp.array4d(np_array_X, dtype=dtype, device="cpu")
    cpu_warp_Y = wp.array4d(np_array_Y, dtype=dtype, device="cpu")

    wp.synchronize()

    gpu_warp_X = wp.array4d(cpu_warp_X, dtype=dtype, device="cuda")
    gpu_warp_Y = wp.array4d(cpu_warp_Y, dtype=dtype, device="cuda")
    wp.synchronize()

    axpy = get_AXPY(f_X=field_X, f_Y=field_Y, alpha=alpha)

    axpy.run(
        stream_idx=0,
        data_view=neon.DataView.standard(),
        container_runtime=container_runtime)

    sk =neon.Skeleton(bk)
    sk.sequence('AXPY', [axpy])
    sk.ioToDot('axpy.dot', 'axpy')

    wp.synchronize()
    wp.launch(
        warp_AXPY,
        dim=(dim.x, dim.y, dim.z),
        inputs=[
            gpu_warp_X,
            gpu_warp_Y,
            alpha
        ]
    )

    wp.synchronize()

    field_Y.update_host(0)
    wp.copy(cpu_warp_Y, gpu_warp_Y)

    wp.synchronize()
    np_array_Y = cpu_warp_Y.numpy()

    for zi in range(0, dim.z):
        for yi in range(0, dim.y):
            for xi in range(0, dim.x):
                for c in range(0, num_card):
                    idx = neon.Index_3d(xi, yi, zi)
                    _, _, expected = golden_axpy(idx, c)
                    neon_res = field_Y.read(idx=idx,
                                            cardinality=c)
                    wp_res = np_array_Y[c, zi, yi, xi]

                    if wp_res != expected:
                        print(f'warp error at {xi},{yi},{zi} :{expected} cvs {wp_res}')
                    assert wp_res == expected

                    if neon_res != expected:
                        print(f'neon error at {xi},{yi},{zi} :{expected} cvs {neon_res}')
                    assert expected == neon_res
    print("Test Passed")




def gpu1_int(dimx, neon_ngpus: int = 1):
    execution(nun_devs=neon_ngpus, num_card=1, dim=neon.Index_3d(1, 1, dimx), dtype=wp.int32,
              container_runtime=neon.Container.ContainerRuntime.neon)


def gpu1_float(dimx, neon_ngpus: int = 1):
    execution(nun_devs=neon_ngpus, num_card=1, dim=neon.Index_3d(dimx, dimx, dimx), dtype=wp.float32,
              container_runtime=neon.Container.ContainerRuntime.neon)


# def gpu1_float():
#     execution(nun_devs=1, num_card=1, dim=neon.Index_3d(10, 10, 10), dtype=ctypes.c_float,
#               container_runtime=neon.Container.ContainerRuntime.neon)

if __name__ == "__main__":
    # gpu1_int()
    # gpu1_int()
    gpu1_int(100, 2)
