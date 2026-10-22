"""Field initialization kernels (Warp) for Carbon grids.

These give ``DenseField`` a Neon-style ``zero_run`` / ``fill_run`` without a
native round-trip: a per-dtype Warp kernel writes ``value`` into every component
of every owned cell, launched once per device over the Standard data view.

This module deliberately does *not* use ``from __future__ import annotations``:
Warp resolves kernel parameter types from concrete annotation objects, so the
partition / span types must be real classes in scope at decoration time.
"""

import warp as wp

from .warp_types import (
    CarbonBlockPartitionF32,
    CarbonBlockPartitionF64,
    CarbonBlockPartitionI32,
    CarbonBlockPartitionI64,
    CarbonBlockSpan,
    CarbonDensePartitionF32,
    CarbonDensePartitionF64,
    CarbonDensePartitionI32,
    CarbonDensePartitionI64,
    CarbonDensePartitionU8,
    CarbonDenseSpan,
)


@wp.kernel
def _dense_fill_f32(
    part: CarbonDensePartitionF32,
    span: CarbonDenseSpan,
    value: wp.float32,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _dense_fill_f64(
    part: CarbonDensePartitionF64,
    span: CarbonDenseSpan,
    value: wp.float64,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _dense_fill_i32(
    part: CarbonDensePartitionI32,
    span: CarbonDenseSpan,
    value: wp.int32,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _dense_fill_i64(
    part: CarbonDensePartitionI64,
    span: CarbonDenseSpan,
    value: wp.int64,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _dense_fill_u8(
    part: CarbonDensePartitionU8,
    span: CarbonDenseSpan,
    value: wp.uint8,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


_DENSE_FILL_KERNELS = {
    "f32": _dense_fill_f32,
    "f64": _dense_fill_f64,
    "i32": _dense_fill_i32,
    "i64": _dense_fill_i64,
    "u8": _dense_fill_u8,
}


@wp.kernel
def _dense_copy_f32(
    dst: CarbonDensePartitionF32,
    src: CarbonDensePartitionF32,
    span: CarbonDenseSpan,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _dense_copy_f64(
    dst: CarbonDensePartitionF64,
    src: CarbonDensePartitionF64,
    span: CarbonDenseSpan,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _dense_copy_i32(
    dst: CarbonDensePartitionI32,
    src: CarbonDensePartitionI32,
    span: CarbonDenseSpan,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _dense_copy_i64(
    dst: CarbonDensePartitionI64,
    src: CarbonDensePartitionI64,
    span: CarbonDenseSpan,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _dense_copy_u8(
    dst: CarbonDensePartitionU8,
    src: CarbonDensePartitionU8,
    span: CarbonDenseSpan,
    ncomp: int,
):
    ix, iy, iz = wp.tid()
    cell = wp.carbon.set(span, ix, iy, iz)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


_DENSE_COPY_KERNELS = {
    "f32": _dense_copy_f32,
    "f64": _dense_copy_f64,
    "i32": _dense_copy_i32,
    "i64": _dense_copy_i64,
    "u8": _dense_copy_u8,
}


# --- Block/Mgrid per-level field ops -------------------------------------------------
# Block-shaped iteration (iter_block_count x MemX x MemY x MemZ) with an active-voxel
# guard, over CarbonBlockPartition + CarbonBlockSpan. Used per level by MgridField
# zero_run / fill_run / copy_from_run (and reusable by Block fields).


@wp.kernel
def _block_fill_f32(part: CarbonBlockPartitionF32, span: CarbonBlockSpan, value: wp.float32, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _block_fill_f64(part: CarbonBlockPartitionF64, span: CarbonBlockSpan, value: wp.float64, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _block_fill_i32(part: CarbonBlockPartitionI32, span: CarbonBlockSpan, value: wp.int32, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


@wp.kernel
def _block_fill_i64(part: CarbonBlockPartitionI64, span: CarbonBlockSpan, value: wp.int64, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(part, span, cell, c, value)


_BLOCK_FILL_KERNELS = {
    "f32": _block_fill_f32,
    "f64": _block_fill_f64,
    "i32": _block_fill_i32,
    "i64": _block_fill_i64,
}


@wp.kernel
def _block_copy_f32(dst: CarbonBlockPartitionF32, src: CarbonBlockPartitionF32, span: CarbonBlockSpan, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _block_copy_f64(dst: CarbonBlockPartitionF64, src: CarbonBlockPartitionF64, span: CarbonBlockSpan, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _block_copy_i32(dst: CarbonBlockPartitionI32, src: CarbonBlockPartitionI32, span: CarbonBlockSpan, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


@wp.kernel
def _block_copy_i64(dst: CarbonBlockPartitionI64, src: CarbonBlockPartitionI64, span: CarbonBlockSpan, ncomp: int):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        wp.carbon.write(dst, span, cell, c, wp.carbon.read(src, span, cell, c))


_BLOCK_COPY_KERNELS = {
    "f32": _block_copy_f32,
    "f64": _block_copy_f64,
    "i32": _block_copy_i32,
    "i64": _block_copy_i64,
}


@wp.kernel
def _block_scatter_f32(part: CarbonBlockPartitionF32, span: CarbonBlockSpan, ncomp: int, out: wp.array4d(dtype=wp.float32)):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    gx = wp.carbon.global_x(span, b, x, y, z)
    gy = wp.carbon.global_y(span, b, x, y, z)
    gz = wp.carbon.global_z(span, b, x, y, z)
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        out[c, gx, gy, gz] = wp.carbon.read(part, span, cell, c)


@wp.kernel
def _block_scatter_f64(part: CarbonBlockPartitionF64, span: CarbonBlockSpan, ncomp: int, out: wp.array4d(dtype=wp.float64)):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    gx = wp.carbon.global_x(span, b, x, y, z)
    gy = wp.carbon.global_y(span, b, x, y, z)
    gz = wp.carbon.global_z(span, b, x, y, z)
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        out[c, gx, gy, gz] = wp.carbon.read(part, span, cell, c)


@wp.kernel
def _block_scatter_i32(part: CarbonBlockPartitionI32, span: CarbonBlockSpan, ncomp: int, out: wp.array4d(dtype=wp.int32)):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    gx = wp.carbon.global_x(span, b, x, y, z)
    gy = wp.carbon.global_y(span, b, x, y, z)
    gz = wp.carbon.global_z(span, b, x, y, z)
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        out[c, gx, gy, gz] = wp.carbon.read(part, span, cell, c)


@wp.kernel
def _block_scatter_i64(part: CarbonBlockPartitionI64, span: CarbonBlockSpan, ncomp: int, out: wp.array4d(dtype=wp.int64)):
    iter_idx, x, y, z = wp.tid()
    b = wp.carbon.view_block_idx(span, iter_idx)
    if not wp.carbon.is_active(span, b, x, y, z):
        return
    gx = wp.carbon.global_x(span, b, x, y, z)
    gy = wp.carbon.global_y(span, b, x, y, z)
    gz = wp.carbon.global_z(span, b, x, y, z)
    cell = wp.carbon.set(span, b, x, y, z)
    for c in range(ncomp):
        out[c, gx, gy, gz] = wp.carbon.read(part, span, cell, c)


_BLOCK_SCATTER_KERNELS = {
    "f32": _block_scatter_f32,
    "f64": _block_scatter_f64,
    "i32": _block_scatter_i32,
    "i64": _block_scatter_i64,
}


def mgrid_to_numpy(field, level: int):
    """Gather every active cell at ``level`` from all devices into a host array.

    Returns an array of shape ``(components, gx, gy, gz)`` (component-major,
    matching XLB's Warp field convention) where ``(gx, gy, gz)`` is the level's
    global voxel extent. Each device scatters its owned active cells into a
    full-extent zero array on its own device; the per-device host arrays are then
    summed (device z-slabs are disjoint, so the sum is a merge) — correct across
    multiple GPUs with no cross-device access.
    """
    import numpy as np

    suffix = field.builtin_suffix
    try:
        kernel = _BLOCK_SCATTER_KERNELS[suffix]
    except KeyError as exc:
        raise ValueError(f"mgrid_to_numpy: unsupported dtype suffix {suffix!r}") from exc

    ncomp = int(field.components)
    gx, gy, gz = field.mgrid.level_extent(level)
    np_dtype = wp.dtype_to_numpy(field.dtype)
    out = np.zeros((ncomp, gx, gy, gz), dtype=np_dtype)

    n_devices = field.gate.lib.carbonpy_device_count()
    for gpu in range(n_devices):
        span, part = _mgrid_level_span(field, level, gpu)
        dim = (int(span.iter_block_count), int(span.mem_x), int(span.mem_y), int(span.mem_z))
        if dim[0] <= 0:
            continue
        device = wp.get_device(f"cuda:{gpu}")
        dev_out = wp.zeros((ncomp, gx, gy, gz), dtype=field.dtype, device=device)
        wp.launch(kernel, dim=dim, inputs=[part, span, ncomp, dev_out], device=device)
        wp.synchronize_device(device)
        out += dev_out.numpy()
    return out


def _mgrid_level_span(field, level: int, gpu: int):
    span = field.mgrid.get_span(level, gpu, 0)
    part = field.get_partition(level, gpu, 0)
    return span, part


def mgrid_fill(field, value, level: int) -> None:
    """Write ``value`` into every component of every active cell of ``field`` at ``level``."""
    suffix = field.builtin_suffix
    try:
        kernel = _BLOCK_FILL_KERNELS[suffix]
    except KeyError as exc:
        raise ValueError(f"mgrid_fill: unsupported dtype suffix {suffix!r}") from exc

    ncomp = int(field.components)
    n_devices = field.gate.lib.carbonpy_device_count()
    devices = []
    for gpu in range(n_devices):
        span, part = _mgrid_level_span(field, level, gpu)
        dim = (int(span.iter_block_count), int(span.mem_x), int(span.mem_y), int(span.mem_z))
        if dim[0] <= 0:
            continue
        device = wp.get_device(f"cuda:{gpu}")
        devices.append(device)
        wp.launch(kernel, dim=dim, inputs=[part, span, field.dtype(value), ncomp], device=device)
    for device in devices:
        wp.synchronize_device(device)


def mgrid_copy(dst, src, level: int) -> None:
    """Copy every component of every active cell at ``level`` from ``src`` into ``dst``."""
    if dst.builtin_suffix != src.builtin_suffix:
        raise ValueError(f"mgrid_copy: dtype mismatch {dst.builtin_suffix!r} != {src.builtin_suffix!r}")
    if int(dst.components) != int(src.components):
        raise ValueError(f"mgrid_copy: component mismatch {int(dst.components)} != {int(src.components)}")
    suffix = dst.builtin_suffix
    try:
        kernel = _BLOCK_COPY_KERNELS[suffix]
    except KeyError as exc:
        raise ValueError(f"mgrid_copy: unsupported dtype suffix {suffix!r}") from exc

    ncomp = int(dst.components)
    n_devices = dst.gate.lib.carbonpy_device_count()
    devices = []
    for gpu in range(n_devices):
        span, dst_part = _mgrid_level_span(dst, level, gpu)
        src_part = src.get_partition(level, gpu, 0)
        dim = (int(span.iter_block_count), int(span.mem_x), int(span.mem_y), int(span.mem_z))
        if dim[0] <= 0:
            continue
        device = wp.get_device(f"cuda:{gpu}")
        devices.append(device)
        wp.launch(kernel, dim=dim, inputs=[dst_part, src_part, span, ncomp], device=device)
    for device in devices:
        wp.synchronize_device(device)


def dense_fill(field, value) -> None:
    """Write ``value`` into every component of every owned cell of ``field``."""
    suffix = field.builtin_suffix
    try:
        kernel = _DENSE_FILL_KERNELS[suffix]
    except KeyError as exc:
        raise ValueError(f"dense_fill: unsupported dtype suffix {suffix!r}") from exc

    ncomp = int(field.components)
    n_devices = field.gate.lib.carbonpy_device_count()
    devices = []
    for gpu in range(n_devices):
        span = field.dense.get_span(gpu, 0)
        dim = (int(span.iter_x), int(span.iter_y), int(span.iter_z))
        if dim[0] <= 0 or dim[1] <= 0 or dim[2] <= 0:
            continue
        part = field.get_partition(gpu, 0)
        device = wp.get_device(f"cuda:{gpu}")
        devices.append(device)
        wp.launch(
            kernel,
            dim=dim,
            inputs=[part, span, field.dtype(value), ncomp],
            device=device,
        )
    for device in devices:
        wp.synchronize_device(device)


def dense_copy(dst, src) -> None:
    """Copy every component of every owned cell from ``src`` into ``dst``.

    Both fields must share dtype, component count, and grid (same per-device
    span). Owned cells only — halos are re-established by a halo update.
    """
    if dst.builtin_suffix != src.builtin_suffix:
        raise ValueError(
            f"dense_copy: dtype mismatch {dst.builtin_suffix!r} != {src.builtin_suffix!r}"
        )
    if int(dst.components) != int(src.components):
        raise ValueError(
            f"dense_copy: component mismatch {int(dst.components)} != {int(src.components)}"
        )
    suffix = dst.builtin_suffix
    try:
        kernel = _DENSE_COPY_KERNELS[suffix]
    except KeyError as exc:
        raise ValueError(f"dense_copy: unsupported dtype suffix {suffix!r}") from exc

    ncomp = int(dst.components)
    n_devices = dst.gate.lib.carbonpy_device_count()
    devices = []
    for gpu in range(n_devices):
        span = dst.dense.get_span(gpu, 0)
        dim = (int(span.iter_x), int(span.iter_y), int(span.iter_z))
        if dim[0] <= 0 or dim[1] <= 0 or dim[2] <= 0:
            continue
        device = wp.get_device(f"cuda:{gpu}")
        devices.append(device)
        wp.launch(
            kernel,
            dim=dim,
            inputs=[dst.get_partition(gpu, 0), src.get_partition(gpu, 0), span, ncomp],
            device=device,
        )
    for device in devices:
        wp.synchronize_device(device)
