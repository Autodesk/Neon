"""XLB single-res grid on the Carbon backend: init + create_field + zero/fill + read-back.

Run inside the CUDA-12 container via scripts/xlb-test-cuda12.sh scripts/_xlb_grid_smoke.py.
Exercises the config/grid rewire (ComputeBackend.NEON -> Carbon) end to end.
"""

import warp as wp

import xlb
from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.precision_policy import PrecisionPolicy


def main() -> None:
    velocity_set = xlb.velocity_set.D3Q19(
        precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.NEON
    )
    xlb.init(
        velocity_set=velocity_set,
        default_backend=ComputeBackend.NEON,
        default_precision_policy=PrecisionPolicy.FP32FP32,
    )

    grid = grid_factory((8, 8, 8), compute_backend=ComputeBackend.NEON, velocity_set=velocity_set)
    store = DefaultConfig.default_precision_policy.store_precision

    rho = grid.create_field(cardinality=1, dtype=store)  # zero-init
    u = grid.create_field(cardinality=3, dtype=store, fill_value=2.0)
    print("created fields: rho(card=1), u(card=3, fill=2.0)")

    gate = DefaultConfig.carbon_gate

    @wp.kernel
    def check(part: gate.CarbonDensePartition, expect: wp.float32, out: wp.array(dtype=wp.int32)):
        ix, iy, iz = wp.tid()
        idx = wp.carbon.set(part, ix, iy, iz)
        for c in range(3):
            if wp.neon_read(part, idx, c) != expect:
                out[0] = 1

    span = grid.grid.get_span(0, 0)
    nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
    bad = wp.zeros(1, dtype=wp.int32, device="cuda:0")
    wp.launch(check, dim=(nx, ny, nz), inputs=[u.get_partition(0, 0), wp.float32(2.0), bad], device="cuda:0")
    wp.synchronize()
    assert int(bad.numpy()[0]) == 0, "u fill_value read-back mismatch"

    bad0 = wp.zeros(1, dtype=wp.int32, device="cuda:0")

    @wp.kernel
    def check0(part: gate.CarbonDensePartition, out: wp.array(dtype=wp.int32)):
        ix, iy, iz = wp.tid()
        idx = wp.carbon.set(part, ix, iy, iz)
        if wp.neon_read(part, idx, 0) != wp.float32(0.0):
            out[0] = 1

    wp.launch(check0, dim=(nx, ny, nz), inputs=[rho.get_partition(0, 0), bad0], device="cuda:0")
    wp.synchronize()
    assert int(bad0.numpy()[0]) == 0, "rho zero_run read-back mismatch"

    print("XLB single-res grid on Carbon: create_field + zero/fill + neon_read OK")


if __name__ == "__main__":
    main()
