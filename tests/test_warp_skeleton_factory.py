#!/usr/bin/env python3
"""Skeleton.sequence with @carbon.kernel loading-lambda kernels."""

import os
import sys
import typing
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpSkeletonFactory(unittest.TestCase):
    def test_skeleton_sequence_updates_field(self) -> None:
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
        from carbon.skeleton import Skeleton  # noqa: WPS433
        from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        dense = carbon.Dense(system, 4, 4, 4)
        field = dense.new_field(1, dtype=wp.float32)
        span = dense.get_span()
        part = field.get_partition()

        @wp.kernel
        def seed(part: CarbonDensePartition, span: CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(part, span, idx, 0, wp.float32(ix + iy + iz))

        wp.launch(
            seed,
            dim=(int(span.iter_x), int(span.iter_y), int(span.iter_z)),
            inputs=[part, span],
            device="cuda:0",
        )
        wp.synchronize()

        driver = CudaDriver(system)

        def _loading(field):
            def setup(loader: carbon.Loader):
                loader.set_grid(field.get_grid())
                f = loader.get_read_write_handle(field)
                s = loader.get_span()

                @wp.func
                def add_kernel(idx: typing.Any):
                    value = wp.carbon.read(f, s, idx, 0)
                    g = wp.carbon.global_idx(s, idx)
                    extra = wp.float32(
                        wp.carbon.get_x(g)
                        + wp.carbon.get_y(g)
                        + wp.carbon.get_z(g)
                    )
                    wp.carbon.write(f, s, idx, 0, value + extra)

                loader.declare_kernel(add_kernel)

            return setup

        op = carbon.build_bound_kernel(_loading(field), driver=driver)
        skeleton = Skeleton(system)
        skeleton.add(op)
        skeleton.run(data_view=0)
        wp.synchronize()

        op.delete()
        driver.delete()
        wp.synchronize()

        out = wp.zeros(
            (int(span.iter_x), int(span.iter_y), int(span.iter_z)),
            dtype=wp.float32,
            device="cuda:0",
        )

        @wp.kernel
        def copy_out(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            arr: wp.array(dtype=wp.float32, ndim=3),
        ):
            ix, iy, iz = wp.tid()
            idx = wp.carbon.set(span, ix, iy, iz)
            arr[ix, iy, iz] = wp.carbon.read(part, span, idx, 0)

        wp.launch(
            copy_out,
            dim=(int(span.iter_x), int(span.iter_y), int(span.iter_z)),
            inputs=[part, span, out],
            device="cuda:0",
        )
        wp.synchronize()

        for iz in range(int(span.iter_z)):
            for iy in range(int(span.iter_y)):
                for ix in range(int(span.iter_x)):
                    base = float(ix + iy + iz)
                    self.assertAlmostEqual(float(out.numpy()[ix, iy, iz]), base * 2.0, places=5)

        skeleton.delete()
        field.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
