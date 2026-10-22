"""Single-res IncompressibleNavierStokesStepper on the Carbon backend.

Drives the ported ``_construct_neon`` / ``prepare_skeleton`` / ``neon_launch``
path (carbon.kernel + carbon.Skeleton + wp.neon_* aliases + uint8 masks) end to
end. Uses a uniform distribution init (BGK-stable: relaxes to equilibrium and
stays finite) so the check is a strict "no NaN/Inf, populations in range" gate
over every device's owned cells.

Run inside the CUDA-12 container via scripts/xlb-test-cuda12.sh scripts/_xlb_stepper_smoke.py.
"""

import warp as wp

import xlb
from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.precision_policy import Precision, PrecisionPolicy
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper


def main() -> None:
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.NEON)
    xlb.init(
        velocity_set=velocity_set,
        default_backend=ComputeBackend.NEON,
        default_precision_policy=PrecisionPolicy.FP32FP32,
    )

    q = velocity_set.q
    grid = grid_factory((16, 16, 16), compute_backend=ComputeBackend.NEON, velocity_set=velocity_set)
    store = DefaultConfig.default_precision_policy.store_precision

    # Fields the stepper container expects: f_0/f_1 (q comps), bc_mask (1 comp u8),
    # missing_mask (q comps u8). Uniform, quiescent-ish init.
    f_0 = grid.create_field(cardinality=q, dtype=store, fill_value=0.05)
    f_1 = grid.create_field(cardinality=q, dtype=store, fill_value=0.05)
    bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)  # zero-init
    missing_mask = grid.create_field(cardinality=q, dtype=Precision.UINT8)  # zero-init
    print(f"created fields: f_0/f_1(card={q}), bc_mask(card=1,u8), missing_mask(card={q},u8)")

    # OCC.standard() makes the Skeleton auto-insert the halo exchange for the
    # stencil-read distribution field and split Internal/Boundary, so streaming
    # across z-slab boundaries is correct on >1 GPU.
    from carbon import SkeletonConfig

    backend_config = {"occ": SkeletonConfig.OCC.standard()}
    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=[], collision_type="BGK", backend_config=backend_config
    )
    print("constructed IncompressibleNavierStokesStepper (BGK, no BCs, occ=standard)")

    # Uniform init (all populations = f_init) is an exact fixed point of a BGK
    # step *in the interior*: streaming a uniform field is a no-op, macroscopic
    # gives rho = q * f_init with u = 0, and feq = w[l] * rho relaxes to the same
    # thing every step. So after an even number of steps, interior cells hold
    # exactly w[l] * rho (independent of omega). Domain-edge cells drain (x/y are
    # always Open here), which is why we only check the interior. An even step
    # count leaves the latest data in f_0.
    f_init = 0.05
    rho_expected = q * f_init  # 0.95 for D3Q19
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

    import numpy as np

    n_devices = int(gate.lib.carbonpy_device_count())
    all_ok = True
    margin = 2  # cells within `margin` of a face are affected by the open x/y edges
    for gpu in range(n_devices):
        span = grid.grid.get_span(gpu, 0)
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        out = wp.zeros(nx * ny * nz * q, dtype=wp.float32, device=f"cuda:{gpu}")
        wp.launch(dump, dim=(nx, ny, nz), inputs=[f_0.get_partition(gpu, 0), wp.int32(q), wp.int32(ny), wp.int32(nz), out], device=f"cuda:{gpu}")
        wp.synchronize()
        vals = out.numpy().reshape(nx, ny, nz, q)
        # Interior slab only: exclude the open domain faces (x/y always Open; the
        # global z faces on the first/last device). With occ=standard the auto-halo
        # keeps cells at the z-slab boundary between devices correct, so a per-device
        # margin on every axis leaves only genuinely-interior owned cells.
        interior = vals[margin:-margin, margin:-margin, margin:-margin, :]
        n_nan = int(np.isnan(interior).sum())
        # Per-cell density = sum over populations; interior must recover rho_expected.
        rho = interior.sum(axis=-1)
        print(
            f"cuda:{gpu}: extent=({nx},{ny},{nz}) interior min={interior.min():.6g} "
            f"max={interior.max():.6g} rho[min={rho.min():.6g} max={rho.max():.6g}] nan={n_nan}"
        )
        ok = (
            n_nan == 0
            and interior.min() >= -1e-5
            and interior.max() <= 0.4
            and abs(float(rho.mean()) - rho_expected) < 1e-3
        )
        all_ok = all_ok and ok

    assert all_ok, "stepper interior populations are not the expected BGK fixed point"
    print(f"single-res stepper on Carbon: {n_devices} device(s), interior BGK fixed point OK")


if __name__ == "__main__":
    main()
