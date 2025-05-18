import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import os
import warp as wp
import neon
import typing


@neon.Container.factory(name='set')
def set(field, level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        if level + 1 < field.get_grid().get_num_levels():
            f = loader.get_mres_write_handle(field, operation=neon.Loader.Operation.stencil_up)
        else:
            f = loader.get_mres_write_handle(field, operation=neon.Loader.Operation.map )

        @wp.func
        def device(cell: typing.Any):
            # wp.neon_print(f_read)
            # get cell global idx
            cartesian_idx = wp.neon_global_idx(f, cell)
            for c in range(wp.neon_cardinality(f)):
                # add the level to each index component
                val = wp.neon_get_component(cartesian_idx, c)
                wp.neon_write(f, cell, c, val)

        loader.declare_kernel(device)

    return kernel

@neon.Container.factory(name='add_level')
def add_level(field, level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)

        f = loader.get_mres_write_handle(field)

        @wp.func
        def device(cell: typing.Any):
            # wp.neon_print(f_read)
            # get cell global idx
            for c in range(wp.neon_cardinality(f)):
                # add the level to each index component
                val = wp.neon_read(f, cell, c)
                val = val + level
                wp.neon_write(f, cell, c, val)

        loader.declare_kernel(device)

    return kernel

@neon.Container.factory(name='copy')
def copy_op(field_in, field_out, level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field_in.get_grid(), level=level)

        f_in = loader.get_mres_read_handle(field_in)
        f_out = loader.get_mres_write_handle(field_out)

        @wp.func
        def device(cell: typing.Any):
            # wp.neon_print(f_read)
            # get cell global idx
            for c in range(wp.neon_cardinality(f_in)):
                # add the level to each index component

                val = wp.neon_read(f_in, cell, c)
                wp.neon_write(f_out, cell, c, val)

        loader.declare_kernel(device)

    return kernel

@neon.Container.factory(name='check')
def test(field_in, field_out, level):
    def kernel(loader: neon.Loader):
        loader.set_mres_grid(field_in.get_grid(), level=level)

        f_in = loader.get_mres_read_handle(field_in)
        f_out = loader.get_mres_write_handle(field_out)

        @wp.func
        def device(cell: typing.Any):
            # wp.neon_print(f_read)
            # get cell global idx
            cartesian_idx = wp.neon_global_idx(f_in, cell)
            for c in range(wp.neon_cardinality(f_in)):
                # add the level to each index component
                expected_val = wp.neon_get_component(cartesian_idx, c)
                expected_val = expected_val + level
                in_val = wp.neon_read(f_in, cell, c)
                out_val = 1
                if in_val != expected_val:
                    out_val = -1
                wp.neon_write(f_out, cell, c, out_val)

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

    grid_shape = (64, 64, 64)
    dim = neon.Index_3d(grid_shape[0],
                        grid_shape[1],
                        grid_shape[2])

    def get_peeled_np(level, width):
        def peel(dim, idx, peel_level, outwards):
            if outwards:
                xIn = idx.x <= peel_level or idx.x >= dim.x - 1 - peel_level
                yIn = idx.y <= peel_level or idx.y >= dim.y - 1 - peel_level
                zIn = idx.z <= peel_level or idx.z >= dim.z - 1 - peel_level
                return xIn or yIn or zIn
            else:
                xIn = idx.x >= peel_level and idx.x <= dim.x - 1 - peel_level
                yIn = idx.y >= peel_level and idx.y <= dim.y - 1 - peel_level
                zIn = idx.z >= peel_level and idx.z <= dim.z - 1 - peel_level
                return xIn and yIn and zIn

        divider = 2 ** level
        m = neon.Index_3d(dim.x // divider, dim.y // divider, dim.z // divider)
        if level == 0:
            m = dim

        mask = np.zeros((m.x, m.y, m.z), dtype=int)
        mask = np.ascontiguousarray(mask, dtype=np.int32)
        # loop over all the elements in mask and set to one any that have x=0 or y=0 or z=0
        for i in range(m.x):
            for j in range(m.y):
                for k in range(m.z):
                    idx = neon.Index_3d(i, j, k)
                    val = 0
                    if peel(m, idx, m.x / width, True):
                        val = 1
                    mask[i, j, k] = val
        return mask

    num_levels = 4
    levels = []

    l0 = get_peeled_np(0, 17)
    l1 = get_peeled_np(1, 7)
    l2 = get_peeled_np(2, 4)
    lastLevel = num_levels - 1
    divider = 2 ** lastLevel
    m = neon.Index_3d(dim.x // divider + 1, dim.y // divider + 1, dim.z // divider + 1)
    lastLevel = np.ones((m.x, m.y, m.z), dtype=int)
    lastLevel = np.ascontiguousarray(lastLevel, dtype=np.int32)
    levels = [l0, l1, l2, lastLevel]

    bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                      dev_idx_list=[0])

    grid = neon.mGrid(bk, dim,
                      sparsity_pattern_list=levels,
                      sparsity_pattern_origins=[neon.Index_3d(0, 0, 0)] * len(levels),
                      stencil=[[0, 0, 0], [1, 0, 0]], )

    print(grid)
    field_a = grid.new_field(cardinality=3, dtype=wp.int32)
    field_b = grid.new_field(cardinality=3, dtype=wp.int32)

    print("Field created")

    wp.synchronize()
    app = []
    for l in range(num_levels):
        app.append(set(field_a, level=l))
    # for l in range(num_levels):
    #     app.append(add_level(field_a, level=l))
    # for l in range(num_levels):
    #     app.append(copy_op(field_a, field_b, level=l))
    # for l in range(num_levels):
    #     app.append(test(field_b, field_a, level=l))

    sk = neon.Skeleton(backend=bk)
    sk.sequence("skeletonTest", app)
    sk.run()
    sk.run()
    sk.run()

    wp.synchronize()
    field_a.update_host(stream=0)
    wp.synchronize()

    field_a.export_vti("mres_skeleton_field_a", "test")


if __name__ == "__main__":
    # block until getting an input from keyboard
    pid = os.getpid()
    print(f"Process PID: {pid}")
    print("Press any key to continue...")
    # input()
    block_grid_try()
