#!/usr/bin/env python3
"""Token-graph skeleton schedules independent sibling containers on separate streams."""

import ctypes
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpSkeletonTokenGraph(unittest.TestCase):
    def test_parallel_sibling_streams(self) -> None:
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
        from carbon.loader import AccessMode  # noqa: WPS433
        from carbon.skeleton import Skeleton  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        field_a = dense.new_field(1, dtype=wp.float32)
        field_b = dense.new_field(1, dtype=wp.float32)
        driver = CudaDriver(system)

        @wp.func
        def fill_a(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ):
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(ix) + wp.float32(1.0)
            )

        @wp.func
        def fill_b(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ):
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(iy) + wp.float32(2.0)
            )

        @wp.func
        def combine_ab(
            a_part: gate.CarbonDensePartition,
            a_span: gate.CarbonDenseSpan,
            b_part: gate.CarbonDensePartition,
            b_span: gate.CarbonDenseSpan,
            ix: int,
            iy: int,
            iz: int,
        ):
            a_cell = wp.carbon.set(a_span, ix, iy, iz)
            b_cell = wp.carbon.set(b_span, ix, iy, iz)
            av = wp.carbon.read(a_part, a_span, a_cell, 0)
            bv = wp.carbon.read(b_part, b_span, b_cell, 0)
            wp.carbon.write(
                a_part, a_span, a_cell, 0, av + bv
            )

        def fill_a_loader(loader, bind_field):
            bind_field(field_a)
            loader.write(field_a)
            loader.declare_kernel(fill_a)

        def fill_b_loader(loader, bind_field):
            bind_field(field_b)
            loader.write(field_b)
            loader.declare_kernel(fill_b)

        def combine_loader(loader, bind_field):
            bind_field(field_a, 0)
            loader.read(field_a)
            bind_field(field_b, 1)
            loader.read(field_b)
            loader.write(field_a)
            loader.declare_kernel(combine_ab)

        fill_a = DenseContainer(driver, dense, fill_a_loader, name="fill_a")
        fill_b = DenseContainer(driver, dense, fill_b_loader, name="fill_b")
        combine_c = DenseContainer(
            driver, dense, combine_loader, name="combine", field_count=2
        )

        for gpu in range(gate.lib.carbonpy_device_count()):
            fill_a.bind_field(field_a, gpu)
            fill_b.bind_field(field_b, gpu)
            combine_c.bind_field(field_a, gpu, field_slot=0)
            combine_c.bind_field(field_b, gpu, field_slot=1)

        sk = Skeleton(system)
        sk.add(fill_a)
        sk.add(fill_b)
        sk.add(combine_c)
        sk.compile()
        streams = sk.kernel_streams()
        self.assertEqual(streams, [0, 1, 0])
        sk.run(data_view=0)
        wp.synchronize()

        span = dense.get_span()
        part = field_a.get_partition()
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
            want = wp.float32(ix) + wp.float32(1.0) + wp.float32(iy) + wp.float32(2.0)
            if got != want:
                out[0] = 1

        wp.launch(check, dim=(nx, ny, nz), inputs=[part, span, mismatches], device="cuda:0")
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        sk.delete()
        combine_c.delete()
        fill_b.delete()
        fill_a.delete()
        driver.delete()
        field_b.delete()
        field_a.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
