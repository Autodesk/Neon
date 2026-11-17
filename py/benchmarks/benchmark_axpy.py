from sympy.abc import alpha

from env_setup import update_pythonpath
import nvtx
update_pythonpath()

import os
import warp as wp
import neon
import typing
from typing import Any

# Generate a function to parte the cli arguments
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="AXPY")
    #parser.add_argument("--help", type=bool, default=False)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="int")
    parser.add_argument("--cardinality", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=1)

    # conver type name to type object
    def type_name(name):
        if name == "int":
            return wp.int32
        if name == "float":
            return wp.float32
        if name == "double":
            return wp.float64
        if name == "int32":
            return wp.int32
        if name == "float32":
            return wp.float32
        if name == "float64":
            return wp.float64
        raise ValueError(f"Unknown type name: {name}")

    setup = parser.parse_args()
    setup.dtype = type_name(setup.dtype)

    return setup

# wp.config.print_launches = True
@wp.kernel
def warp_AXPY(
        x: wp.array4d(dtype=Any),
        y: wp.array4d(dtype=Any),
        alpha: Any):
    i, j, k = wp.tid()
    for c in range(x.shape[0]):
        y[c, k, j, i] = x[c, k, j, i] + alpha * y[c, k, j, i]


@neon.Container.factory(name = "AXPY")
def get_AXPY(f_X, f_Y, alpha_: Any):
    def axpy(loader: neon.Loader):
        loader.set_grid(f_Y.get_grid())

        f_x = loader.get_read_handle(f_X)
        f_y = loader.get_read_handle(f_Y)
        alpha = f_X.dtype(2)
        max_c = wp.int32(f_X.get_cardinality())
        # try:
        #     max_c = int(max_c)
        # except:
        #     pass
        @wp.func
        def foo(idx: typing.Any):
            for c in range(max_c):
                x = wp.neon_read(f_x, idx, c)
                y = wp.neon_read(f_y, idx, c)
                axpy_res = x + alpha * y
                wp.neon_write(f_y, idx, c, axpy_res)
        loader.declare_kernel(foo)

    return axpy

@neon.Container.factory()
def set_to_random(f_X, f_Y):
    def axpy(loader: neon.Loader):
        loader.set_grid(f_Y.get_grid())

        f_x = loader.get_read_handle(f_X)
        f_y = loader.get_read_handle(f_Y)
        max_c = int(f_X.get_cardinality())
        dtype = f_X.dtype

        seed = 42

        @wp.func
        def generate_random(g: typing.Any):
            a = dtype(1.22)
            b = dtype(4.33)
            return a, b

        @wp.func
        def foo(idx: typing.Any):
            for c in range(max_c):
                global_idx = wp.neon_global_idx(f_x, idx)
                r1 , r2 = generate_random(global_idx)

                wp.neon_write(f_x, idx, c, r1)
                wp.neon_write(f_y, idx, c, r2)


        loader.declare_kernel(foo)

    return axpy


def execution_axpy(repetitions_id ,
                   nun_devs: int,
                   num_card: int,
                   dim: neon.Index_3d,
                   dtype,
                   container_runtime: neon.Container.ContainerRuntime,
                   iterations=100,
                   ):


    dev_idx_list = list(range(nun_devs))
    bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                      dev_idx_list=dev_idx_list)

    grid = neon.dense.dGrid(bk, dim, sparsity=None, stencil=[[0, 0, 0]])
    field_X = grid.new_field(cardinality=num_card, dtype=dtype)
    field_Y = grid.new_field(cardinality=num_card, dtype=dtype)
    field_result = grid.new_field(cardinality=num_card, dtype=dtype)


    #field_X.update_device(0)
    #field_Y.update_device(0)
    #field_result.update_device(0)

    nvtx.push_range("Init data", color="red")

    init_data = set_to_random(f_X=field_X, f_Y=field_Y)

    init_data.run(stream_idx=0,
                  data_view=neon.DataView.standard(),
                  container_runtime=container_runtime)

    wp.synchronize()
    nvtx.pop_range()

    aalpha = dtype(1.0)
    axpy_even = get_AXPY(f_X=field_X, f_Y=field_Y, alpha_=1)
    axpy_odd = get_AXPY( f_X=field_Y,f_Y=field_X, alpha_=aalpha)

    nvtx.push_range("AXPY - warmup")

    # loop for 4 times
    for i in range(0, 4):
        axpy_odd.run(
            stream_idx=0,
            data_view=neon.DataView.standard(),
            container_runtime=container_runtime)

        axpy_even.run(
            stream_idx=0,
            data_view=neon.DataView.standard(),
            container_runtime=container_runtime)

    nvtx.pop_range()


    #start nvtx section

    # sync up start timer
    bk.sync()
    timer = neon.Timer()
    timer.start()
    nvtx.push_range("AXPY - benchmark")

    # loop for iterations times
    for i in range(0, iterations):
        axpy_odd.run(
            stream_idx=0,
            data_view=neon.DataView.standard(),
            container_runtime=container_runtime)

        axpy_even.run(
            stream_idx=0,
            data_view=neon.DataView.standard(),
            container_runtime=container_runtime)
    # bk.info();

    bk.sync()
    enlapsed_time_ms = timer.stop()
    # compute million lattice update per second (MLUPS)
    mlups = (dim.x * dim.y * dim.z * iterations) / enlapsed_time_ms / 1000

    #stop nvtx section
    nvtx.pop_range()
    return mlups, enlapsed_time_ms


def benchmark(dim, ngpus: int = 1. ,dtype=wp.float32, iterations=100, repetitions=1):
    # conver dtype to string
    def type_name(dtype):
        if dtype == wp.int32:
            return "int32"
        if dtype == wp.float32:
            return "float32"
        if dtype == wp.float64:
            return "float64"
        raise ValueError(f"Unknown type: {dtype}")

    # Create a neon report and add the parameters of the benchmark


    report = neon.Report()
    report.add_member('dim', dim)
    report.add_member('ngpus', ngpus)
    report.add_member('dtype', type_name(dtype))
    report.add_member('iterations', iterations)
    report.add_member('repetitions', repetitions)
    report.add_member('runtime', 'py_neon')


    # loop over repetitions
    mlups = []
    enlapsed_time_ms = []
    for i in range(repetitions):
        mlups_i, time_i = execution_axpy(repetitions_id=i,
                                         nun_devs=ngpus,
                                         num_card=1,
                                         dim=neon.Index_3d(dim, dim, dim),
                                         dtype=dtype,
                                         container_runtime=neon.Container.ContainerRuntime.neon, iterations=100)
        mlups.append(mlups_i)
        enlapsed_time_ms.append(time_i)
        print(f"MLUPS: {mlups_i}")
    print(f"MLUPS: {mlups}")

    report.add_member_vector('mlups', mlups)
    report.add_member_vector('time (ms)', enlapsed_time_ms)

    report_name = f"axpy_py_neon_{dim}_{ngpus}_{type_name(dtype)}_{iterations}_{repetitions}.json"
    report.write(report_name)


if __name__ == "__main__":
    import os

    current_pid = os.getpid()
    print(f"The PID of this script is: {current_pid}")

    # Get the path of the current script
    script_path = __file__
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    wp.config.mode = "release"
    wp.config.llvm_cuda = False
    wp.config.verbose = False
    wp.verbose_warnings = False

    wp.init()
    neon.init()
    # Parse the cli arguments
    args = parse_args()
    print(f"Arguments: {args}")
    benchmark(args.dim, args.ngpus, args.dtype, args.iterations, args.repetitions)
    pass
