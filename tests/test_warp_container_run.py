#!/usr/bin/env python3
"""Dense container run via CUDA Driver API + Warp CUfunction hooks."""

import ctypes
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpContainerRun(unittest.TestCase):
    def test_dense_container_fill(self) -> None:
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
        from carbon.container import CudaDriver, DenseContainer  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(1, dtype=wp.float32)
        driver = CudaDriver(system)

        @wp.func
        def fill_cell(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ):
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(ix + iy + iz)
            )

        def fill_loader(loader, bind_field):
            bind_field(field)
            loader.declare_kernel(fill_cell)

        container = DenseContainer(driver, dense, fill_loader)
        for gpu in range(gate.lib.carbonpy_device_count()):
            container.bind_field(field, gpu)
        container.run(data_view=0)
        wp.synchronize()

        span = dense.get_span()
        part = field.get_partition()
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
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

        container.delete()
        driver.delete()
        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
