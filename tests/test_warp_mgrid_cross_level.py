#!/usr/bin/env python3
"""mGrid cross-level parent/child reads via CarbonMgridPartition Warp builtins."""

import ctypes
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpMgridCrossLevel(unittest.TestCase):
    def test_cross_level_parent_child_reads(self) -> None:
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

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        mgrid = carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=2)
        field = mgrid.new_field(1, dtype=wp.float32)
        depth = mgrid.depth
        ref_factor = 2

        @wp.func
        def encode(level: int, x: int, y: int, z: int):
            return wp.float32(level * 1000000 + (x + 100 * y + 10000 * z))

        @wp.kernel
        def fill_level(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            level: int,
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            gx = wp.carbon.global_x(span, b, x, y, z)
            gy = wp.carbon.global_y(span, b, x, y, z)
            gz = wp.carbon.global_z(span, b, x, y, z)
            cell = wp.carbon.set(span, b, x, y, z)
            wp.carbon.write(
                part, span, cell, 0, encode(level, gx, gy, gz)
            )

        for level in range(depth):
            span = mgrid.get_span(level)
            part = field.get_partition(level)
            wp.launch(
                fill_level,
                dim=(
                    int(span.iter_block_count),
                    int(span.mem_x),
                    int(span.mem_y),
                    int(span.mem_z),
                ),
                inputs=[part, span, level],
                device="cuda:0",
            )
        wp.synchronize()

        err = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        ok = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check_children(
            part: gate.CarbonMgridPartition,
            own_span: gate.CarbonBlockSpan,
            rf: int,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(own_span, iter_idx)
            if not wp.carbon.is_active(own_span, b, x, y, z):
                return
            if not wp.carbon.has_children(part, b, x, y, z):
                wp.atomic_add(err_count, 0, 1)
                return
            for oz in range(rf):
                for oy in range(rf):
                    for ox in range(rf):
                        if not wp.carbon.child_valid(part, b, x, y, z, ox, oy, oz):
                            wp.atomic_add(err_count, 0, 1)
                            continue
                        cell = wp.carbon.set(own_span, b, x, y, z)
                        got = wp.carbon.read_child(part, cell, ox, oy, oz, 0)
                        gx = wp.carbon.child_global_x(part, b, x, y, z, ox, oy, oz)
                        gy = wp.carbon.child_global_y(part, b, x, y, z, ox, oy, oz)
                        gz = wp.carbon.child_global_z(part, b, x, y, z, ox, oy, oz)
                        want = encode(0, gx, gy, gz)
                        if got != want:
                            wp.atomic_add(err_count, 0, 1)
                        else:
                            wp.atomic_add(ok_count, 0, 1)

        @wp.kernel
        def check_parents(
            part: gate.CarbonMgridPartition,
            own_span: gate.CarbonBlockSpan,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(own_span, iter_idx)
            if not wp.carbon.is_active(own_span, b, x, y, z):
                return
            if not wp.carbon.has_parent(part, b):
                wp.atomic_add(err_count, 0, 1)
                return
            if not wp.carbon.parent_valid(part, own_span, b, x, y, z):
                wp.atomic_add(err_count, 0, 1)
                return
            cell = wp.carbon.set(own_span, b, x, y, z)
            got = wp.carbon.read(part, own_span, cell, 0)
            gx = wp.carbon.parent_global_x(part, own_span, b, x, y, z)
            gy = wp.carbon.parent_global_y(part, own_span, b, x, y, z)
            gz = wp.carbon.parent_global_z(part, own_span, b, x, y, z)
            want = encode(1, gx, gy, gz)
            if got != want:
                wp.atomic_add(err_count, 0, 1)
            else:
                wp.atomic_add(ok_count, 0, 1)

        level = 1
        own_span = mgrid.get_span(level)
        cross_part = field.get_cross_partition(level)
        wp.launch(
            check_children,
            dim=(
                int(own_span.iter_block_count),
                int(own_span.mem_x),
                int(own_span.mem_y),
                int(own_span.mem_z),
            ),
            inputs=[cross_part, own_span, ref_factor, err, ok],
            device="cuda:0",
        )

        level = 0
        own_span = mgrid.get_span(level)
        cross_part = field.get_cross_partition(level)
        wp.launch(
            check_parents,
            dim=(
                int(own_span.iter_block_count),
                int(own_span.mem_x),
                int(own_span.mem_y),
                int(own_span.mem_z),
            ),
            inputs=[cross_part, own_span, err, ok],
            device="cuda:0",
        )
        wp.synchronize()

        err_count = int(err.numpy()[0])
        ok_count = int(ok.numpy()[0])
        self.assertEqual(err_count, 0, f"cross-level mismatches: err={err_count} ok={ok_count}")
        self.assertGreater(ok_count, 0)

        field.delete()
        mgrid.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
