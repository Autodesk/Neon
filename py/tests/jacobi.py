import neon
from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon as ne
from neon import Index_3d
from neon import Ngh_idx
from neon.dense import dSpan
import typing
from typing import Any


@wp.kernel
def warp_jacobi(
        x: wp.array3d(dtype=Any),
        y: wp.array3d(dtype=Any),
        dim: wp.vec3i):
    i, j, k = wp.tid()
    dx = dim[0]
    dy = dim[1]
    dz = dim[2]
    tmp = wp.float(0)
    offsets = wp.vec2i(-1, 1)
    for di_ in range(2):
        for dj_ in range(2):
            for dk_ in range(2):
                di = offsets[di_]
                dj = offsets[dj_]
                dk = offsets[dk_]
                if (0 <= i + di < dx and
                        0 <= j + dj < dy and
                        0 <= k + dk < dz):
                    tmp = tmp + x[k + dk, j + dj, i + di]
    y[k, j, i] = tmp / wp.float(6)


@ne.Container.factory()
def get_jacobi(f_X, f_Y):
    def axpy(loader: ne.Loader):
        loader.set_grid(f_Y.get_grid())

        f_x = loader.get_read_handle(f_X)
        f_y = loader.get_read_handle(f_Y)

        @wp.func
        def foo(idx: typing.Any):
            #            wp.neon_print(idx)
            # wp.neon_print(f_read)
            tmp = wp.float(0)
            offsets = wp.vec2i(-1, 1)
            for di_ in range(2):
                for dj_ in range(2):
                    for dk_ in range(2):
                        di = offsets[di_]
                        dj = offsets[dj_]
                        dk = offsets[dk_]

                        ngh = wp.neon_ngh_idx(wp.int8(di),
                                                 wp.int8(dj),
                                                 wp.int8(dk))
                        unused_is_valid = wp.bool(False)
                        tmp = tmp + wp.neon_read_ngh(f_x,
                                                     idx,
                                                     ngh,
                                                     wp.int32(0),
                                                     wp.float(0),
                                                     unused_is_valid)
            wp.neon_write(f_y, idx, 0, tmp / wp.float(6))

        loader.declare_kernel(foo)

    return axpy


def execution(nun_devs: int,
              dim: ne.Index_3d,
              dtype,
              container_runtime: ne.Container.ContainerRuntime):
    num_card = 1
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
    ne.init()

    dev_idx_list = list(range(nun_devs))
    bk = ne.Backend(runtime=ne.Backend.Runtime.stream,
                    dev_idx_list=dev_idx_list)

    grid = ne.dense.dGrid(bk, dim)
    field_X = grid.new_field(cardinality=num_card, dtype=dtype)
    field_Y = grid.new_field(cardinality=num_card, dtype=dtype)
    field_result = grid.new_field(cardinality=num_card, dtype=dtype)

    # Initialize two 4D arrays on the CPU
    cpu_warp_X = wp.empty(shape=(dim.z, dim.y, dim.x), dtype=dtype, device="cpu")
    cpu_warp_Y = wp.empty(shape=(dim.z, dim.y, dim.x), dtype=dtype, device="cpu")
    gpu_warp_X = wp.empty(shape=(dim.z, dim.y, dim.x), dtype=dtype, device="cuda")
    gpu_warp_Y = wp.empty(shape=(dim.z, dim.y, dim.x), dtype=dtype, device="cuda")

    def golden_jacobi(index: ne.Index_3d):
        def golden_jacobi_input(idx: ne.Index_3d):
            return dtype(idx.x + idx.y + idx.z), dtype(idx.x + idx.y + idx.z + 1)

        tmp = dtype(0)
        for di in [-1, 1]:
            for dj in [-1, 1]:
                for dk in [-1, 1]:
                    ngh = ne.Index_3d(idx.x + di, idx.y + dj, idx.z + dk)
                    if (0 <= ngh.x < dim.x and
                            0 <= ngh.y < dim.y and
                            0 <= ngh.z < dim.z):
                        ngh_val_on_x, _ = golden_jacobi_input(ngh)
                        tmp = tmp + ngh_val_on_x
        x, y = golden_jacobi_input(index)
        return x, y, tmp/wp.float32(6)

    np_array_X = cpu_warp_X.numpy()
    np_array_Y = cpu_warp_Y.numpy()

    for zi in range(0, dim.z):
        for yi in range(0, dim.y):
            for xi in range(0, dim.x):
                idx = Index_3d(xi, yi, zi)
                # print(f"idx {idx}")
                x, y, result = golden_jacobi(idx)
                field_X.write(idx=idx,
                              cardinality=0,
                              newValue=dtype(x))
                field_Y.write(idx=idx,
                              cardinality=0,
                              newValue=dtype(y))
                field_result.write(idx,
                                   cardinality=0,
                                   newValue=dtype(result))

                np_array_X[zi, yi, xi] = x
                np_array_Y[zi, yi, xi] = y

    field_X.update_device(0)
    field_Y.update_device(0)
    field_result.update_device(0)

    cpu_warp_X = wp.array3d(np_array_X, dtype=dtype, device="cpu")
    cpu_warp_Y = wp.array3d(np_array_Y, dtype=dtype, device="cpu")

    wp.synchronize()

    gpu_warp_X = wp.array3d(cpu_warp_X, dtype=dtype, device="cuda")
    gpu_warp_Y = wp.array3d(cpu_warp_Y, dtype=dtype, device="cuda")
    wp.synchronize()

    jacobi = get_jacobi(f_X=field_X, f_Y=field_Y)

    jacobi.run(
        stream_idx=0,
        data_view=ne.DataView.standard(),
        container_runtime=container_runtime)

    wp.synchronize()
    wp.launch(
        warp_jacobi,
        dim=wp.vec3i(dim.x, dim.y, dim.z),
        inputs=[
            gpu_warp_X,
            gpu_warp_Y,
            wp.vec3i(dim.x, dim.y, dim.z)
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
                c = 0
                idx = Index_3d(xi, yi, zi)
                _, _, expected = golden_jacobi(idx)
                computed = field_Y.read(idx=idx,
                                        cardinality=0)
                if expected != computed:
                    print(f'neon error at {xi},{yi},{zi} :{expected} cvs {computed}')
                assert expected == computed

                wp_res = np_array_Y[zi, yi, xi]
                if wp_res != expected:
                    print(f'warp error at {xi},{yi},{zi} :{expected} cvs {wp_res}')
                assert expected == computed
    print("Test Passed")



def gpu1_int(dimx, neon_ngpus: int = 1):
    execution(nun_devs=neon_ngpus, dim=ne.Index_3d(dimx, dimx, dimx), dtype=int,
              container_runtime=ne.Container.ContainerRuntime.neon)


def gpu1_float(dimx, neon_ngpus: int = 1):
    execution(nun_devs=neon_ngpus, dim=ne.Index_3d(dimx, dimx, dimx), dtype=wp.float32,
              container_runtime=ne.Container.ContainerRuntime.neon)


# def gpu1_float():
#     execution(nun_devs=1, num_card=1, dim=ne.Index_3d(10, 10, 10), dtype=ctypes.c_float,
#               container_runtime=ne.Container.ContainerRuntime.neon)

if __name__ == "__main__":
    # gpu1_int()
    # gpu1_int()
    gpu1_float(10, 1)
