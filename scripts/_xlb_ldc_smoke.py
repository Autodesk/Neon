"""Lid-driven-cavity end-to-end smoke on the Carbon backend (public name: NEON).

Mirrors examples/performance/mlups_3d.py's setup on a small grid: an
EquilibriumBC moving lid on top + FullwayBounceBackBC walls, BGK collision,
the IndicesBoundaryMasker, prepare_fields(), and the fused nse_stepper under
carbon.Skeleton with occ=standard. Validates that the masker tagged the
boundary, the sim stays finite/bounded, and the lid drives interior flow
(max u_x > 0). This is the first test to exercise boundary conditions on
Carbon end-to-end.

Run: scripts/xlb-test-cuda12.sh scripts/_xlb_ldc_smoke.py
"""

import numpy as np
import warp as wp

import xlb
from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from carbon import SkeletonConfig
from carbon.warp_types import CarbonDensePartitionU8


def main() -> None:
    lid_speed = 0.05
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.NEON)
    xlb.init(
        velocity_set=velocity_set,
        default_backend=ComputeBackend.NEON,
        default_precision_policy=PrecisionPolicy.FP32FP32,
    )

    grid_shape = (24, 24, 24)
    options = {"occ": SkeletonConfig.OCC.standard(), "device_list": [0]}
    grid = grid_factory(grid_shape, compute_backend=ComputeBackend.NEON, velocity_set=velocity_set, backend_config=options)

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    boundary_conditions = [
        EquilibriumBC(rho=1.0, u=(lid_speed, 0.0, 0.0), indices=lid),
        FullwayBounceBackBC(indices=walls),
    ]

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="BGK",
        backend_config=options,
    )

    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
    wp.synchronize()
    print("prepare_fields() OK (masker + BCs + copy_from_run + initialize_eq)")

    gate = DefaultConfig.carbon_gate
    q = velocity_set.q
    n_devices = int(gate.lib.carbonpy_device_count())

    # --- verify the masker tagged the boundary (bc_mask nonzero somewhere) ---
    @wp.kernel
    def count_tagged(part: CarbonDensePartitionU8, span: gate.CarbonDenseSpan, out: wp.array(dtype=wp.int32)):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(span, ix, iy, iz)
        if wp.carbon.read(part, span, cell, 0) != wp.uint8(0):
            wp.atomic_add(out, 0, 1)

    @wp.kernel
    def hist(part: CarbonDensePartitionU8, span: gate.CarbonDenseSpan, out: wp.array(dtype=wp.int32)):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(span, ix, iy, iz)
        v = wp.int32(wp.carbon.read(part, span, cell, 0))
        wp.atomic_add(out, v, 1)

    # --- probe: does neon_global_idx/get_{x,y,z} agree with (ix,iy,iz) on every plane? ---
    @wp.kernel
    def probe(part: CarbonDensePartitionU8, span: gate.CarbonDenseSpan, gx: wp.array(dtype=wp.int32), gy: wp.array(dtype=wp.int32), gz: wp.array(dtype=wp.int32), ny: wp.int32, nz: wp.int32):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(span, ix, iy, iz)
        g = wp.neon_global_idx(part, cell)
        off = (ix * ny + iy) * nz + iz
        gx[off] = wp.neon_get_x(g)
        gy[off] = wp.neon_get_y(g)
        gz[off] = wp.neon_get_z(g)

    span0 = grid.grid.get_span(0, 0)
    nx0, ny0, nz0 = int(span0.iter_x), int(span0.iter_y), int(span0.iter_z)
    gxa = wp.zeros(nx0 * ny0 * nz0, dtype=wp.int32, device="cuda:0")
    gya = wp.zeros(nx0 * ny0 * nz0, dtype=wp.int32, device="cuda:0")
    gza = wp.zeros(nx0 * ny0 * nz0, dtype=wp.int32, device="cuda:0")
    wp.launch(probe, dim=(nx0, ny0, nz0), inputs=[bc_mask.get_partition(0, 0), span0, gxa, gya, gza, wp.int32(ny0), wp.int32(nz0)], device="cuda:0")
    wp.synchronize()
    GX = gxa.numpy().reshape(nx0, ny0, nz0)
    GY = gya.numpy().reshape(nx0, ny0, nz0)
    GZ = gza.numpy().reshape(nx0, ny0, nz0)
    ii, jj, kk = np.meshgrid(np.arange(nx0), np.arange(ny0), np.arange(nz0), indexing="ij")
    print(f"probe gx==ix: {bool((GX == ii).all())}  gy==iy: {bool((GY == jj).all())}  gz==iz: {bool((GZ == kk).all())}")
    for attr in ("iter_x", "iter_y", "iter_z", "logic_x", "logic_y", "logic_z", "halo_z", "z_offset", "global_z", "pitch_xy", "data_view"):
        if hasattr(span0, attr):
            print(f"  span0.{attr} = {getattr(span0, attr)}")
    print(f"  gz column at (ix=0,iy=0): {GZ[0, 0, :].tolist()}")

    total_tagged = 0
    hist_total = np.zeros(256, dtype=np.int64)
    for gpu in range(n_devices):
        span = grid.grid.get_span(gpu, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        cnt = wp.zeros(1, dtype=wp.int32, device=f"cuda:{gpu}")
        h = wp.zeros(256, dtype=wp.int32, device=f"cuda:{gpu}")
        wp.launch(count_tagged, dim=(nx, ny, nz), inputs=[bc_mask.get_partition(gpu, 0), span, cnt], device=f"cuda:{gpu}")
        wp.launch(hist, dim=(nx, ny, nz), inputs=[bc_mask.get_partition(gpu, 0), span, h], device=f"cuda:{gpu}")
        wp.synchronize()
        total_tagged += int(cnt.numpy()[0])
        hist_total += h.numpy().astype(np.int64)
    lid_id = boundary_conditions[0].id
    wall_id = boundary_conditions[1].id
    print(f"bc ids: lid(EquilibriumBC)={lid_id} wall(FullwayBounceBack)={wall_id}")
    print(f"expected: lid cells={len(lid[0])} wall cells={len(walls[0])}")
    print(f"bc_mask hist: id={lid_id}->{hist_total[lid_id]}, id={wall_id}->{hist_total[wall_id]}, total tagged={total_tagged}")
    assert total_tagged > 0, "masker tagged no boundary cells"

    # --- run the LDC with growth diagnostics ---
    omega = 1.0
    rho = grid.create_field(cardinality=1, dtype=PrecisionPolicy.FP32FP32.store_precision)
    u = grid.create_field(cardinality=3, dtype=PrecisionPolicy.FP32FP32.store_precision)
    macro = Macroscopic(
        compute_backend=ComputeBackend.NEON,
        precision_policy=PrecisionPolicy.FP32FP32,
        velocity_set=velocity_set,
    )

    @wp.kernel
    def maxabs(part: gate.CarbonDensePartition, out: wp.array(dtype=wp.float32)):
        ix, iy, iz = wp.tid()
        idx = wp.carbon.set(part, ix, iy, iz)
        v = wp.abs(wp.float32(wp.neon_read(part, idx, 0)))
        wp.atomic_max(out, 0, v)

    def report(tag, cur):
        macro(cur, rho, u)
        wp.synchronize()
        m = 0.0
        for gpu in range(n_devices):
            span = grid.grid.get_span(gpu, 0)
            nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
            mm = wp.zeros(1, dtype=wp.float32, device=f"cuda:{gpu}")
            wp.launch(maxabs, dim=(nx, ny, nz), inputs=[u.get_partition(gpu, 0), mm], device=f"cuda:{gpu}")
            wp.synchronize()
            m = max(m, float(mm.numpy()[0]))
        print(f"  [{tag}] max|u_x| = {m:.6g}")

    report("init", f_0)
    checkpoints = {1, 2, 5, 10, 20, 50}
    n_steps = 50
    for t in range(n_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, t)
        f_0, f_1 = f_1, f_0
        if (t + 1) in checkpoints:
            report(f"step {t + 1}", f_0)
    wp.synchronize()
    print(f"ran {n_steps} LDC steps via carbon.Skeleton")

    macro(f_0, rho, u)
    wp.synchronize()

    @wp.kernel
    def dump(u_part: gate.CarbonDensePartition, rho_part: gate.CarbonDensePartition, ux: wp.array(dtype=wp.float32), rho: wp.array(dtype=wp.float32), ny: wp.int32, nz: wp.int32):
        ix, iy, iz = wp.tid()
        idx = wp.carbon.set(u_part, ix, iy, iz)
        off = (ix * ny + iy) * nz + iz
        ux[off] = wp.float32(wp.neon_read(u_part, idx, 0))
        rho[off] = wp.float32(wp.neon_read(rho_part, idx, 0))

    margin = 2
    ok = True
    global_uxi_max = -1e30
    for gpu in range(n_devices):
        span = grid.grid.get_span(gpu, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        ux_a = wp.zeros(nx * ny * nz, dtype=wp.float32, device=f"cuda:{gpu}")
        rho_a = wp.zeros(nx * ny * nz, dtype=wp.float32, device=f"cuda:{gpu}")
        wp.launch(dump, dim=(nx, ny, nz), inputs=[u.get_partition(gpu, 0), rho.get_partition(gpu, 0), ux_a, rho_a, wp.int32(ny), wp.int32(nz)], device=f"cuda:{gpu}")
        wp.synchronize()
        ux = ux_a.numpy().reshape(nx, ny, nz)
        rr = rho_a.numpy().reshape(nx, ny, nz)
        uxi = ux[margin:-margin, margin:-margin, margin:-margin]
        rri = rr[margin:-margin, margin:-margin, margin:-margin]
        n_nan = int(np.isnan(ux).sum() + np.isnan(rr).sum())
        print(
            f"cuda:{gpu}: extent=({nx},{ny},{nz}) full u_x[{ux.min():.4g},{ux.max():.4g}] "
            f"interior u_x[{uxi.min():.4g},{uxi.max():.4g}] rho[{rri.min():.4g},{rri.max():.4g}] nan={n_nan}"
        )
        ok = ok and n_nan == 0 and uxi.max() <= 0.2 and uxi.min() >= -0.2
        ok = ok and rri.min() > 0.5 and rri.max() < 1.5
        global_uxi_max = max(global_uxi_max, float(uxi.max()))

    # Lid drives positive x-flow somewhere in the interior (strongest near the top slab;
    # far-from-lid slabs are legitimately near-quiescent, so this is a global check).
    ok = ok and global_uxi_max > 1e-3
    assert ok, "LDC interior velocity/density is not finite/bounded or lid did not drive flow"
    print(f"LDC on Carbon: {n_devices} device(s), masker+BCs+stepper all OK, interior stable, lid drives flow")


if __name__ == "__main__":
    main()
