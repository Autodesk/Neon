#!/usr/bin/env python3
"""Neighbour / read_ngh builtins for Dense, Block, and Mgrid."""

import os
import sys
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


class CarbonWarpNeighbour(unittest.TestCase):
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

    def test_dense_neighbour_interior_and_boundary(self) -> None:
        import warp as wp

        system = self.carbon.System(self.gate)
        dense = self.carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(1, dtype=wp.float32)
        span = dense.get_span()
        part = field.get_partition()
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        if nx < 2:
            self.skipTest("device-0 slab too thin in x for a +x neighbour probe")
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        expected = float((cx + 1) + cy + cz)

        @wp.kernel
        def seed(part: self.CarbonDensePartition, span: self.CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(part, span, cell, 0, wp.float32(ix + iy + iz))

        wp.launch(seed, dim=(nx, ny, nz), inputs=[part, span], device="cuda:0")
        wp.synchronize()

        interior = wp.zeros(2, dtype=wp.float32, device="cuda:0")
        boundary = wp.zeros(1, dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def check(
            part: self.CarbonDensePartition,
            span: self.CarbonDenseSpan,
            interior_out: wp.array(dtype=wp.float32),
            boundary_out: wp.array(dtype=wp.float32),
            cx: int,
            cy: int,
            cz: int,
        ):
            cell = wp.carbon.set(span, cx, cy, cz)
            xp = wp.carbon.neighbour(part, span, cell, 1, 0, 0, 0)
            interior_out[0] = wp.carbon.value(xp)
            interior_out[1] = wp.float32(1.0) if wp.carbon.valid(xp) else wp.float32(0.0)

            edge = wp.carbon.set(span, 0, 0, 0)
            xm = wp.carbon.read_ngh(part, span, edge, -1, 0, 0, 0, wp.float32(-1.0))
            boundary_out[0] = xm

        wp.launch(
            check,
            dim=1,
            inputs=[part, span, interior, boundary, cx, cy, cz],
            device="cuda:0",
        )
        wp.synchronize()

        self.assertAlmostEqual(float(interior.numpy()[0]), expected, places=5)
        self.assertAlmostEqual(float(interior.numpy()[1]), 1.0, places=5)
        self.assertAlmostEqual(float(boundary.numpy()[0]), -1.0, places=5)

        field.delete()
        dense.delete()
        system.delete()

    def test_block_neighbour_active_voxel(self) -> None:
        import warp as wp

        system = self.carbon.System(self.gate)
        block = self.carbon.Block(system, 16, 16, 16)
        field = block.new_field(1, dtype=wp.float32)
        span = block.get_span()
        part = field.get_partition()

        @wp.kernel
        def seed(part: self.CarbonBlockPartition, span: self.CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            cell = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, cell):
                return
            wp.carbon.write(part, span, cell, 0, wp.float32(x + y + z))

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

        out = wp.zeros(2, dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def check(
            part: self.CarbonBlockPartition,
            span: self.CarbonBlockSpan,
            arr: wp.array(dtype=wp.float32),
        ):
            iter_idx, x, y, z = wp.tid()
            if iter_idx != 0 or x != 4 or y != 4 or z != 4:
                return
            cell = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, cell):
                return
            xp = wp.carbon.neighbour(part, span, cell, 1, 0, 0, 0)
            arr[0] = wp.carbon.value_or(xp, wp.float32(-1.0))
            arr[1] = wp.float32(1.0) if wp.carbon.valid(xp) else wp.float32(0.0)

        wp.launch(
            check,
            dim=(
                int(span.iter_block_count),
                int(span.mem_x),
                int(span.mem_y),
                int(span.mem_z),
            ),
            inputs=[part, span, out],
            device="cuda:0",
        )
        wp.synchronize()

        self.assertAlmostEqual(float(out.numpy()[0]), 13.0, places=5)
        self.assertAlmostEqual(float(out.numpy()[1]), 1.0, places=5)

        field.delete()
        block.delete()
        system.delete()

    def test_mgrid_neighbour_same_level(self) -> None:
        import warp as wp

        system = self.carbon.System(self.gate)
        mgrid = self.carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=2)
        field = mgrid.new_field(1, dtype=wp.float32)
        level = 0
        span = mgrid.get_span(level)
        part = field.get_partition(level)

        @wp.kernel
        def seed(part: self.CarbonBlockPartition, span: self.CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            cell = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, cell):
                return
            wp.carbon.write(part, span, cell, 0, wp.float32(x + y + z))

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

        out = wp.zeros(1, dtype=wp.float32, device="cuda:0")

        @wp.kernel
        def check(
            part: self.CarbonBlockPartition,
            span: self.CarbonBlockSpan,
            arr: wp.array(dtype=wp.float32),
        ):
            iter_idx, x, y, z = wp.tid()
            if iter_idx != 0 or x != 2 or y != 2 or z != 2:
                return
            cell = wp.carbon.set(span, iter_idx, x, y, z)
            if not wp.carbon.is_valid(span, cell):
                return
            arr[0] = wp.carbon.read_ngh(part, span, cell, 0, 1, 0, 0, wp.float32(-1.0))

        wp.launch(
            check,
            dim=(
                int(span.iter_block_count),
                int(span.mem_x),
                int(span.mem_y),
                int(span.mem_z),
            ),
            inputs=[part, span, out],
            device="cuda:0",
        )
        wp.synchronize()

        self.assertAlmostEqual(float(out.numpy()[0]), 7.0, places=5)

        field.delete()
        mgrid.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
