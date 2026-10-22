#!/usr/bin/env python3
"""Dense AXPY via Carbon Warp builtins vs a reference wp.array kernel."""

import ctypes
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpAxpy(unittest.TestCase):
    def test_dense_axpy_matches_reference(self) -> None:
        root = _repo_root()
        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp
        except ImportError:
            self.skipTest("warp module not installed")

        import carbon  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        x_field = dense.new_field(1, dtype=wp.float32)
        y_field = dense.new_field(1, dtype=wp.float32)
        x_span = dense.get_span()
        y_span = dense.get_span()
        x_part = x_field.get_partition()
        y_part = y_field.get_partition()

        alpha = 2.5
        nx, ny, nz = int(x_span.iter_x), int(x_span.iter_y), int(x_span.iter_z)

        @wp.kernel
        def seed(part: gate.CarbonDensePartition, span: gate.CarbonDenseSpan, scale: float):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, scale * wp.float32(ix + iy + iz)
            )

        @wp.kernel
        def copy_from_carbon(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            out: wp.array(dtype=wp.float32, ndim=3),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            out[ix, iy, iz] = wp.carbon.read(part, span, cell, 0)

        @wp.kernel
        def axpy_carbon(
            x_part: gate.CarbonDensePartition,
            x_span: gate.CarbonDenseSpan,
            y_part: gate.CarbonDensePartition,
            y_span: gate.CarbonDenseSpan,
            a: float,
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(y_span, ix, iy, iz)
            x_cell = wp.carbon.set(x_span, ix, iy, iz)
            xv = wp.carbon.read(x_part, x_span, x_cell, 0)
            yv = wp.carbon.read(y_part, y_span, cell, 0)
            wp.carbon.write(
                y_part, y_span, cell, 0, xv + a * yv
            )

        @wp.kernel
        def axpy_ref(x: wp.array(dtype=wp.float32, ndim=3), y: wp.array(dtype=wp.float32, ndim=3), a: float):
            ix, iy, iz = wp.tid()
            y[ix, iy, iz] = x[ix, iy, iz] + a * y[ix, iy, iz]

        wp.launch(seed, dim=(nx, ny, nz), inputs=[x_part, x_span, 1.0], device="cuda:0")
        wp.launch(seed, dim=(nx, ny, nz), inputs=[y_part, y_span, 3.0], device="cuda:0")
        wp.synchronize()

        x_arr = wp.zeros((nx, ny, nz), dtype=wp.float32, device="cuda:0")
        y_arr = wp.zeros((nx, ny, nz), dtype=wp.float32, device="cuda:0")
        wp.launch(copy_from_carbon, dim=(nx, ny, nz), inputs=[x_part, x_span, x_arr], device="cuda:0")
        wp.launch(copy_from_carbon, dim=(nx, ny, nz), inputs=[y_part, y_span, y_arr], device="cuda:0")
        wp.synchronize()

        wp.launch(
            axpy_carbon,
            dim=(nx, ny, nz),
            inputs=[x_part, x_span, y_part, y_span, float(alpha)],
            device="cuda:0",
        )
        wp.launch(axpy_ref, dim=(nx, ny, nz), inputs=[x_arr, y_arr, float(alpha)], device="cuda:0")
        wp.synchronize()

        got = wp.zeros((nx, ny, nz), dtype=wp.float32, device="cuda:0")
        wp.launch(copy_from_carbon, dim=(nx, ny, nz), inputs=[y_part, y_span, got], device="cuda:0")
        wp.synchronize()

        got_np = got.numpy()
        y_np = y_arr.numpy()
        self.assertLess(float(abs(got_np - y_np).max()), 1e-5)

        x_field.delete()
        y_field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
