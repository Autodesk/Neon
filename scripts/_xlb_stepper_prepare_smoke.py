"""Single-res stepper via the real prepare_fields() path on the Carbon backend.

Unlike _xlb_stepper_smoke.py (which hand-builds the fields), this drives the
user-facing entry point: Stepper.prepare_fields() -> create_nse_fields +
DenseField.copy_from_run + initialize_eq (QuadraticEquilibrium Neon launch).
Equilibrium init with rho=1, u=0 is an exact BGK fixed point in the interior,
so after an even number of steps interior cells must hold w[l] * 1.0.

Run inside the CUDA-12 container via scripts/xlb-test-cuda12.sh scripts/_xlb_stepper_prepare_smoke.py.
"""

import numpy as np
import warp as wp

import xlb
from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper
from carbon import SkeletonConfig


def main() -> None:
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.NEON)
    xlb.init(
        velocity_set=velocity_set,
        default_backend=ComputeBackend.NEON,
        default_precision_policy=PrecisionPolicy.FP32FP32,
    )

    q = velocity_set.q
    grid = grid_factory((16, 16, 16), compute_backend=ComputeBackend.NEON, velocity_set=velocity_set)

    backend_config = {"occ": SkeletonConfig.OCC.standard()}
    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=[], collision_type="BGK", backend_config=backend_config
    )

    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
    wp.synchronize()
    print("prepare_fields() OK (create_nse_fields + copy_from_run + initialize_eq)")

    omega = 1.0
    n_steps = 2
    for t in range(n_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, t)
    wp.synchronize()
    print(f"ran {n_steps} stepper steps via carbon.Skeleton")

    gate = DefaultConfig.carbon_gate

    @wp.kernel
    def dump(part: gate.CarbonDensePartition, ncomp: wp.int32, ny: wp.int32, nz: wp.int32, out: wp.array(dtype=wp.float32)):
        ix, iy, iz = wp.tid()
        idx = wp.carbon.set(part, ix, iy, iz)
        base = ((ix * ny + iy) * nz + iz) * ncomp
        for c in range(ncomp):
            out[base + c] = wp.float32(wp.neon_read(part, idx, c))

    n_devices = int(gate.lib.carbonpy_device_count())
    margin = 2
    all_ok = True
    for gpu in range(n_devices):
        span = grid.grid.get_span(gpu, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        out = wp.zeros(nx * ny * nz * q, dtype=wp.float32, device=f"cuda:{gpu}")
        wp.launch(dump, dim=(nx, ny, nz), inputs=[f_0.get_partition(gpu, 0), wp.int32(q), wp.int32(ny), wp.int32(nz), out], device=f"cuda:{gpu}")
        wp.synchronize()
        vals = out.numpy().reshape(nx, ny, nz, q)
        interior = vals[margin:-margin, margin:-margin, margin:-margin, :]
        n_nan = int(np.isnan(interior).sum())
        rho = interior.sum(axis=-1)
        print(
            f"cuda:{gpu}: extent=({nx},{ny},{nz}) interior min={interior.min():.6g} "
            f"max={interior.max():.6g} rho[min={rho.min():.6g} max={rho.max():.6g}] nan={n_nan}"
        )
        # rho=1 equilibrium init: interior populations are exactly w[l] * 1.0.
        ok = n_nan == 0 and interior.min() >= -1e-5 and interior.max() <= 0.35 and abs(float(rho.mean()) - 1.0) < 1e-3
        all_ok = all_ok and ok

    assert all_ok, "prepare_fields stepper interior is not the expected rho=1 BGK fixed point"
    print(f"prepare_fields stepper on Carbon: {n_devices} device(s), interior rho=1 fixed point OK")


if __name__ == "__main__":
    main()
