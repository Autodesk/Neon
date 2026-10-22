#!/usr/bin/env python3
"""One 6-neighbour Jacobi average step on a Dense grid via @carbon.kernel.

Mirrors tests/test_warp_skeleton_halo.py with ``OCC.standard()``.
"""

import typing

import numpy as np

from _carbon_env import require_libcarbonpy, require_warp, setup_paths

setup_paths()


def _jacobi_reference(u: np.ndarray) -> np.ndarray:
    nx, ny, nz = u.shape
    v = np.zeros_like(u)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                acc = 0.0
                if ix > 0:
                    acc += u[ix - 1, iy, iz]
                if ix + 1 < nx:
                    acc += u[ix + 1, iy, iz]
                if iy > 0:
                    acc += u[ix, iy - 1, iz]
                if iy + 1 < ny:
                    acc += u[ix, iy + 1, iz]
                if iz > 0:
                    acc += u[ix, iy, iz - 1]
                if iz + 1 < nz:
                    acc += u[ix, iy, iz + 1]
                v[ix, iy, iz] = acc / 6.0
    return v


def main() -> int:
    root = setup_paths()
    require_libcarbonpy(root)
    wp = require_warp()

    import carbon
    from carbon.loader import Operation
    from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan

    gate = carbon.init(enable_warp=True)
    n_gpus = gate.lib.carbonpy_device_count()
    if n_gpus <= 0:
        print("jacobi_step: no CUDA devices visible")
        return 1

    global_x = global_y = global_z = 8
    system = carbon.System(gate)
    dense = carbon.Dense(system, global_x, global_y, global_z, halo_radius=1)
    u_field = dense.new_field(1, dtype=wp.float32)
    v_field = dense.new_field(1, dtype=wp.float32)

    @wp.kernel
    def seed(part: CarbonDensePartition, span: CarbonDenseSpan, z_origin: int):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(span, ix, iy, iz)
        wp.carbon.write(part, span, cell, 0, wp.float32(ix + iy + z_origin + iz))

    @wp.kernel
    def copy_from_carbon(
        part: CarbonDensePartition,
        span: CarbonDenseSpan,
        out: wp.array(dtype=wp.float32, ndim=3),
    ):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(span, ix, iy, iz)
        out[ix, iy, iz] = wp.carbon.read(part, span, cell, 0)

    for gpu in range(n_gpus):
        u_part = u_field.get_partition(gpu)
        u_span = dense.get_span(gpu)
        iter_x, iter_y, iter_z = int(u_span.iter_x), int(u_span.iter_y), int(u_span.iter_z)
        z_origin = int(u_span.z_offset + u_span.halo_z)
        wp.launch(
            seed,
            dim=(iter_x, iter_y, iter_z),
            inputs=[u_part, u_span, z_origin],
            device=f"cuda:{gpu}",
        )
    wp.synchronize()

    @carbon.kernel
    def jacobi_step(u_field, v_field):
        def setup(loader: carbon.Loader):
            loader.set_grid(u_field.get_grid())
            u = loader.get_read_handle(u_field, operation=Operation.stencil)
            v = loader.get_write_handle(v_field)
            span = loader.get_span()

            @wp.func
            def jacobi_kernel(idx: typing.Any):
                acc = wp.carbon.read_ngh(u, span, idx, -1, 0, 0, 0, wp.float32(0.0))
                acc = acc + wp.carbon.read_ngh(u, span, idx, 1, 0, 0, 0, wp.float32(0.0))
                acc = acc + wp.carbon.read_ngh(u, span, idx, 0, -1, 0, 0, wp.float32(0.0))
                acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 1, 0, 0, wp.float32(0.0))
                acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 0, -1, 0, wp.float32(0.0))
                acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 0, 1, 0, wp.float32(0.0))
                wp.carbon.write(v, span, idx, 0, acc / wp.float32(6.0))

            loader.declare_kernel(jacobi_kernel)

        return setup

    jacobi = jacobi_step(u_field, v_field)
    skel = carbon.Skeleton(system)
    skel.sequence("jacobi", [jacobi], occ=carbon.OCC.standard())
    skel.run()
    wp.synchronize()

    got = np.zeros((global_x, global_y, global_z), dtype=np.float32)
    u_host = np.zeros((global_x, global_y, global_z), dtype=np.float32)
    for gpu in range(n_gpus):
        v_span = dense.get_span(gpu)
        u_span = dense.get_span(gpu)
        iter_x, iter_y, iter_z = int(v_span.iter_x), int(v_span.iter_y), int(v_span.iter_z)
        z_off = int(v_span.z_offset + v_span.halo_z)

        slab = wp.zeros((iter_x, iter_y, iter_z), dtype=wp.float32, device=f"cuda:{gpu}")
        wp.launch(
            copy_from_carbon,
            dim=(iter_x, iter_y, iter_z),
            inputs=[v_field.get_partition(gpu), v_span, slab],
            device=f"cuda:{gpu}",
        )
        got[:, :, z_off : z_off + iter_z] = slab.numpy()

        u_slab = wp.zeros((iter_x, iter_y, iter_z), dtype=wp.float32, device=f"cuda:{gpu}")
        wp.launch(
            copy_from_carbon,
            dim=(iter_x, iter_y, iter_z),
            inputs=[u_field.get_partition(gpu), u_span, u_slab],
            device=f"cuda:{gpu}",
        )
        u_host[:, :, z_off : z_off + iter_z] = u_slab.numpy()
    wp.synchronize()

    ref = _jacobi_reference(u_host)
    diff = float(np.abs(got - ref).max())
    print(f"jacobi_step: gpus={n_gpus} occ=standard max |carbon - reference| = {diff:.3e}")
    ok = diff < 1e-5

    skel.delete()
    jacobi.delete()
    u_field.delete()
    v_field.delete()
    dense.delete()
    system.delete()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
