#!/usr/bin/env python3
"""Neon-style @carbon.kernel loading-lambda kernels for Dense, Block, and Mgrid."""

import os
import sys
import typing
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _skip_unless_warp_ready(root: str):
    lib_path = os.path.join(
        root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
    )
    if not os.path.isfile(lib_path):
        return f"libcarbonpy not built at {lib_path}"
    sys.path.insert(0, os.path.join(root, "py"))
    try:
        import warp as wp  # noqa: F401
    except ImportError:
        return "warp module not installed"
    import warp as wp

    repo = getattr(wp, "__file__", "") or ""
    if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
        return "Warp external-type fork not installed; run ./scripts/setup_warp.sh"
    return None


class CarbonWarpKernelFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = _repo_root()
        reason = _skip_unless_warp_ready(root)
        if reason:
            raise unittest.SkipTest(reason)
        import carbon  # noqa: WPS433
        from carbon.warp_types import (  # noqa: WPS433
            CarbonBlockPartition,
            CarbonBlockSpan,
            CarbonDensePartition,
            CarbonDenseSpan,
        )

        cls.carbon = carbon
        cls.CarbonBlockPartition = CarbonBlockPartition
        cls.CarbonBlockSpan = CarbonBlockSpan
        cls.CarbonDensePartition = CarbonDensePartition
        cls.CarbonDenseSpan = CarbonDenseSpan
        cls.gate = carbon.init(enable_warp=True)
        if cls.gate.lib.carbonpy_device_count() <= 0:
            raise unittest.SkipTest("no CUDA devices visible")

    def test_dense_two_pass_add(self) -> None:
        import warp as wp

        carbon = self.carbon
        system = carbon.System(self.gate)
        dense = carbon.Dense(system, 1, 1, 3)
        field = dense.new_field(1, dtype=wp.float32)
        span = dense.get_span()
        part = field.get_partition()

        @wp.kernel
        def seed(part: self.CarbonDensePartition, span: self.CarbonDenseSpan):
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

        @carbon.kernel
        def add_operator(field):
            def setup(loader: carbon.Loader):
                loader.set_grid(field.get_grid())
                f = loader.get_read_handle(field)
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

        op = add_operator(field)
        op.run()
        op.run()
        wp.synchronize()

        out = wp.zeros(
            (int(span.iter_x), int(span.iter_y), int(span.iter_z)),
            dtype=wp.float32,
            device="cuda:0",
        )

        @wp.kernel
        def copy_out(
            part: self.CarbonDensePartition,
            span: self.CarbonDenseSpan,
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
                    self.assertAlmostEqual(float(out.numpy()[ix, iy, iz]), base * 3.0, places=5)

        op.delete()
        field.delete()
        dense.delete()
        system.delete()

    def test_block_two_pass_add(self) -> None:
        import warp as wp

        carbon = self.carbon
        system = carbon.System(self.gate)
        block = carbon.Block(system, 16, 16, 16)
        field = block.new_field(1, dtype=wp.float32)
        span = block.get_span()
        part = field.get_partition()

        @wp.kernel
        def seed(part: self.CarbonBlockPartition, span: self.CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            idx = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, idx):
                return
            g = wp.carbon.global_idx(span, idx)
            wp.carbon.write(
                part,
                span,
                idx,
                0,
                wp.float32(
                    wp.carbon.get_x(g) + wp.carbon.get_y(g) + wp.carbon.get_z(g)
                ),
            )

        wp.launch(
            seed,
            dim=(
                int(span.iter_block_count),
                int(span.mem_x),
                int(span.mem_y),
                int(span.mem_z),
            ),
            inputs=[part, span],
            device="cuda:0",
        )
        wp.synchronize()

        @carbon.kernel
        def add_operator(field):
            def setup(loader: carbon.Loader):
                loader.set_grid(field.get_grid())
                f = loader.get_read_handle(field)
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

        op = add_operator(field)
        op.run()
        op.run()
        wp.synchronize()

        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(
            part: self.CarbonBlockPartition,
            span: self.CarbonBlockSpan,
            out: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            idx = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, idx):
                return
            g = wp.carbon.global_idx(span, idx)
            want = wp.float32(
                wp.float32(
                    wp.carbon.get_x(g) + wp.carbon.get_y(g) + wp.carbon.get_z(g)
                )
                * wp.float32(3.0)
            )
            if wp.carbon.read(part, span, idx, 0) != want:
                out[0] = 1

        wp.launch(
            check,
            dim=(
                int(span.iter_block_count),
                int(span.mem_x),
                int(span.mem_y),
                int(span.mem_z),
            ),
            inputs=[part, span, mismatches],
            device="cuda:0",
        )
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        op.delete()
        field.delete()
        block.delete()
        system.delete()

    def test_mgrid_two_pass_add(self) -> None:
        import warp as wp

        carbon = self.carbon
        system = carbon.System(self.gate)
        mgrid = carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=2)
        field = mgrid.new_field(1, dtype=wp.float32)
        level = 0
        span = mgrid.get_span(level)
        part = field.get_partition(level)

        @wp.kernel
        def seed(part: self.CarbonBlockPartition, span: self.CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            idx = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, idx):
                return
            g = wp.carbon.global_idx(span, idx)
            wp.carbon.write(
                part,
                span,
                idx,
                0,
                wp.float32(
                    wp.carbon.get_x(g) + wp.carbon.get_y(g) + wp.carbon.get_z(g)
                ),
            )

        wp.launch(
            seed,
            dim=(
                int(span.iter_block_count),
                int(span.mem_x),
                int(span.mem_y),
                int(span.mem_z),
            ),
            inputs=[part, span],
            device="cuda:0",
        )
        wp.synchronize()

        @carbon.kernel
        def add_operator(field):
            def setup(loader: carbon.Loader):
                loader.set_mres_grid(field.get_grid(), level)
                f = loader.get_mres_read_handle(field)
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

        op = add_operator(field)
        op.run()
        op.run()
        wp.synchronize()

        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(
            part: self.CarbonBlockPartition,
            span: self.CarbonBlockSpan,
            out: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            idx = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, idx):
                return
            g = wp.carbon.global_idx(span, idx)
            want = wp.float32(
                wp.float32(
                    wp.carbon.get_x(g) + wp.carbon.get_y(g) + wp.carbon.get_z(g)
                )
                * wp.float32(3.0)
            )
            if wp.carbon.read(part, span, idx, 0) != want:
                out[0] = 1

        wp.launch(
            check,
            dim=(
                int(span.iter_block_count),
                int(span.mem_x),
                int(span.mem_y),
                int(span.mem_z),
            ),
            inputs=[part, span, mismatches],
            device="cuda:0",
        )
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        op.delete()
        field.delete()
        mgrid.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
