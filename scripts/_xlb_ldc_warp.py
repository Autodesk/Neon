"""Same LDC as _xlb_ldc_smoke.py but on the WARP backend (reference path).

Used to distinguish a Carbon-backend bug from shared XLB LBM logic: if WARP is
stable (max|u| ~ lid speed) and NEON diverges, the defect is in Carbon's neon
path; if both diverge, it's in the shared functionals / setup.
"""

import numpy as np
import warp as wp

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic


def main() -> None:
    lid_speed = 0.05
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(velocity_set=velocity_set, default_backend=ComputeBackend.WARP, default_precision_policy=PrecisionPolicy.FP32FP32)

    grid_shape = (24, 24, 24)
    grid = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP, velocity_set=velocity_set)

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    boundary_conditions = [
        EquilibriumBC(rho=1.0, u=(lid_speed, 0.0, 0.0), indices=lid),
        FullwayBounceBackBC(indices=walls),
    ]
    stepper = IncompressibleNavierStokesStepper(grid=grid, boundary_conditions=boundary_conditions, collision_type="BGK")
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
    wp.synchronize()

    rho = grid.create_field(cardinality=1, dtype=PrecisionPolicy.FP32FP32.store_precision)
    u = grid.create_field(cardinality=3, dtype=PrecisionPolicy.FP32FP32.store_precision)
    macro = Macroscopic(compute_backend=ComputeBackend.WARP, precision_policy=PrecisionPolicy.FP32FP32, velocity_set=velocity_set)

    def report(tag, cur):
        macro(cur, rho, u)
        wp.synchronize()
        ux = u.numpy()[0]
        print(f"  [{tag}] max|u_x| = {np.abs(ux).max():.6g}")

    report("init", f_0)
    omega = 1.0
    checkpoints = {1, 2, 5, 10, 20, 50}
    for t in range(50):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, t)
        f_0, f_1 = f_1, f_0
        if (t + 1) in checkpoints:
            report(f"step {t + 1}", f_0)
    print("WARP LDC done")


if __name__ == "__main__":
    main()
