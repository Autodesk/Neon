"""Container factories shared by the tests. Names are prefixed ``NeonTest``."""

import typing

import warp as wp

import neon


# --- dense / block grids -----------------------------------------------------

@neon.Container.factory(name="NeonTestAddCoordSum")
def add_coord_sum(field):
    """field[idx] += idx.x + idx.y + idx.z (cardinality 0)."""

    def setup(loader: neon.Loader):
        loader.set_grid(field.get_grid())
        f = loader.get_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            g = wp.neon_global_idx(f, idx)
            wp.neon_write(f, idx, 0, value + wp.neon_get_x(g) + wp.neon_get_y(g) + wp.neon_get_z(g))

        loader.declare_kernel(kernel)

    return setup


# --- multi-resolution grids --------------------------------------------------

@neon.Container.factory(name="NeonTestMresSetCoordSum")
def mres_set_coord_sum(field, level):
    """field[level, idx, 0] = global_idx.x + global_idx.y + global_idx.z."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            g = wp.neon_global_idx(f, idx)
            wp.neon_write(f, idx, 0, wp.neon_get_x(g) + wp.neon_get_y(g) + wp.neon_get_z(g))

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresAddCoordSum")
def mres_add_coord_sum(field, level):
    """field[level, idx, 0] += global_idx.x + global_idx.y + global_idx.z."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            value = wp.neon_read(f, idx, 0)
            g = wp.neon_global_idx(f, idx)
            wp.neon_write(f, idx, 0, value + wp.neon_get_x(g) + wp.neon_get_y(g) + wp.neon_get_z(g))

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresCountCoordSumMismatch")
def mres_count_coord_sum_mismatch(field, level, factor, errors):
    """errors[0] += 1 for every voxel where field != factor * coord_sum(global_idx)."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_read_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            g = wp.neon_global_idx(f, idx)
            expected = factor * (wp.neon_get_x(g) + wp.neon_get_y(g) + wp.neon_get_z(g))
            if wp.neon_read(f, idx, 0) != expected:
                wp.atomic_add(errors, 0, 1)

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresWriteGlobalIdx")
def mres_write_global_idx(field, level, offset=0, operation=None):
    """field[level, idx, c] = global_idx[c] + offset for every component."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        if operation is None:
            f = loader.get_mres_write_handle(field)
        else:
            f = loader.get_mres_write_handle(field, operation=operation)

        @wp.func
        def kernel(idx: typing.Any):
            g = wp.neon_global_idx(f, idx)
            for c in range(wp.neon_cardinality(f)):
                wp.neon_write(f, idx, c, wp.neon_get_component(g, c) + offset)

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresAddConstant")
def mres_add_constant(field, level, value):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            for c in range(wp.neon_cardinality(f)):
                wp.neon_write(f, idx, c, wp.neon_read(f, idx, c) + value)

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresCopy")
def mres_copy(src, dst, level):
    def setup(loader: neon.Loader):
        loader.set_mres_grid(src.get_grid(), level=level)
        f_in = loader.get_mres_read_handle(src)
        f_out = loader.get_mres_write_handle(dst)

        @wp.func
        def kernel(idx: typing.Any):
            for c in range(wp.neon_cardinality(f_in)):
                wp.neon_write(f_out, idx, c, wp.neon_read(f_in, idx, c))

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresCheckGlobalIdx")
def mres_check_global_idx(src, dst, level, offset):
    """dst[c] = 1 where src[c] == global_idx[c] + offset, else -1."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(src.get_grid(), level=level)
        f_in = loader.get_mres_read_handle(src)
        f_out = loader.get_mres_write_handle(dst)

        @wp.func
        def kernel(idx: typing.Any):
            g = wp.neon_global_idx(f_in, idx)
            for c in range(wp.neon_cardinality(f_in)):
                result = 1
                if wp.neon_read(f_in, idx, c) != wp.neon_get_component(g, c) + offset:
                    result = -1
                wp.neon_write(f_out, idx, c, result)

        loader.declare_kernel(kernel)

    return setup


@neon.Container.factory(name="NeonTestMresTopologyFlags")
def mres_topology_flags(field, level):
    """c0 = has_child, c1 = has_parent (as 0/1). Needs cardinality >= 2."""

    def setup(loader: neon.Loader):
        loader.set_mres_grid(field.get_grid(), level=level)
        f = loader.get_mres_write_handle(field)

        @wp.func
        def kernel(idx: typing.Any):
            has_child = 0
            if wp.neon_has_child(f, idx):
                has_child = 1
            has_parent = 0
            if wp.neon_has_parent(f, idx):
                has_parent = 1
            wp.neon_write(f, idx, 0, has_child)
            wp.neon_write(f, idx, 1, has_parent)

        loader.declare_kernel(kernel)

    return setup
