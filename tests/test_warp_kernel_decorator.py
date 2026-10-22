#!/usr/bin/env python3
"""Dense container kernels via the @dense.kernel decorator vs reference wp.launch."""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpKernelDecorator(unittest.TestCase):
    def test_dense_fill_via_decorator(self) -> None:
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

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        import carbon  # noqa: WPS433
        from carbon.container import CudaDriver  # noqa: WPS433
        from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(1, dtype=wp.float32)
        driver = CudaDriver(system)

        @dense.kernel(driver, fields=[field], access=["write"])
        def fill_cell(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ):
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(ix + iy + iz)
            )

        fill_cell.run(data_view=0)
        wp.synchronize()

        span = dense.get_span()
        part = field.get_partition()
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            got = wp.carbon.read(part, span, cell, 0)
            want = wp.float32(ix + iy + iz)
            if got != want:
                out[0] = 1

        wp.launch(check, dim=(nx, ny, nz), inputs=[part, span, mismatches], device="cuda:0")
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        fill_cell.delete()
        driver.delete()
        field.delete()
        dense.delete()
        system.delete()

    def test_dense_jacobi_via_decorator(self) -> None:
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

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        import carbon  # noqa: WPS433
        from carbon.container import CudaDriver  # noqa: WPS433
        from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        u_field = dense.new_field(1, dtype=wp.float32)
        v_field = dense.new_field(1, dtype=wp.float32)
        u_span = dense.get_span()
        v_span = dense.get_span()
        u_part = u_field.get_partition()
        v_part = v_field.get_partition()
        nx, ny, nz = int(u_span.iter_x), int(u_span.iter_y), int(u_span.iter_z)
        driver = CudaDriver(system)

        @wp.kernel
        def seed(part: CarbonDensePartition, span: CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(ix + iy + iz)
            )

        wp.launch(seed, dim=(nx, ny, nz), inputs=[u_part, u_span], device="cuda:0")
        wp.synchronize()

        @wp.func
        def read_open(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ) -> float:
            if ix < 0 or iy < 0 or iz < 0 or ix >= 8 or iy >= 8 or iz >= 8:
                return wp.float32(0.0)
            cell = wp.carbon.set(span, ix, iy, iz)
            return wp.carbon.read(part, span, cell, 0)

        @dense.kernel(
            driver,
            fields=[u_field, v_field],
            access=["read", "write"],
        )
        def jacobi(
            u_part: CarbonDensePartition,
            u_span: CarbonDenseSpan,
            v_part: CarbonDensePartition,
            v_span: CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ):
            acc = read_open(u_part, u_span, ix - 1, iy, iz)
            acc = acc + read_open(u_part, u_span, ix + 1, iy, iz)
            acc = acc + read_open(u_part, u_span, ix, iy - 1, iz)
            acc = acc + read_open(u_part, u_span, ix, iy + 1, iz)
            acc = acc + read_open(u_part, u_span, ix, iy, iz - 1)
            acc = acc + read_open(u_part, u_span, ix, iy, iz + 1)
            cell = wp.carbon.set(v_span, ix, iy, iz)
            wp.carbon.write(
                v_part, v_span, cell, 0, acc / wp.float32(6.0)
            )

        jacobi.run(data_view=0)
        wp.synchronize()

        u_arr = wp.zeros((nx, ny, nz), dtype=wp.float32, device="cuda:0")
        v_arr = wp.zeros((nx, ny, nz), dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def copy_from_carbon(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            out: wp.array(dtype=wp.float32, ndim=3),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            out[ix, iy, iz] = wp.carbon.read(part, span, cell, 0)

        @wp.kernel
        def jacobi_ref(u: wp.array(dtype=wp.float32, ndim=3), v: wp.array(dtype=wp.float32, ndim=3)):
            ix, iy, iz = wp.tid()
            acc = wp.float32(0.0)
            if ix > 0:
                acc = acc + u[ix - 1, iy, iz]
            if ix + 1 < nx:
                acc = acc + u[ix + 1, iy, iz]
            if iy > 0:
                acc = acc + u[ix, iy - 1, iz]
            if iy + 1 < ny:
                acc = acc + u[ix, iy + 1, iz]
            if iz > 0:
                acc = acc + u[ix, iy, iz - 1]
            if iz + 1 < nz:
                acc = acc + u[ix, iy, iz + 1]
            v[ix, iy, iz] = acc / wp.float32(6.0)

        wp.launch(copy_from_carbon, dim=(nx, ny, nz), inputs=[u_part, u_span, u_arr], device="cuda:0")
        wp.launch(jacobi_ref, dim=(nx, ny, nz), inputs=[u_arr, v_arr], device="cuda:0")
        wp.synchronize()

        got = wp.zeros((nx, ny, nz), dtype=wp.float32, device="cuda:0")
        wp.launch(copy_from_carbon, dim=(nx, ny, nz), inputs=[v_part, v_span, got], device="cuda:0")
        wp.synchronize()

        diff = abs(got.numpy() - v_arr.numpy()).max()
        self.assertLess(float(diff), 1e-5)

        jacobi.delete()
        driver.delete()
        u_field.delete()
        v_field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
