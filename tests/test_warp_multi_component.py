#!/usr/bin/env python3
"""Multi-component block / mGrid addressing.

Locks the invariant that block-family offset math takes its component count from the
*partition*, not from the span. ``carbonpy_block_get_span`` cannot know a field's
component count -- it is per-grid -- so a span-sourced count silently addressed every
multi-component field as if it had one component. Single-component tests cannot see the
difference, so these two exercise components > 1 on the own-level block path and on the
mGrid cross-level parent / child paths.
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpMultiComponent(unittest.TestCase):
    def _bootstrap(self):
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
        return wp, carbon, gate

    def test_block_components(self) -> None:
        wp, carbon, gate = self._bootstrap()
        ncomp = 3

        system = carbon.System(gate)
        block = carbon.Block(system, 16, 16, 16)
        field = block.new_field(ncomp, dtype=wp.float32)

        @wp.func
        def encode(b: int, x: int, y: int, z: int, c: int):
            return wp.float32((((b * 8 + x) * 8 + y) * 8 + z) * 4 + c)

        @wp.kernel
        def fill(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            comps: int,
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(span, b, x, y, z)
            for c in range(comps):
                wp.carbon.write(part, span, cell, c, encode(b, x, y, z, c))

        @wp.kernel
        def check(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            comps: int,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(span, b, x, y, z)
            for c in range(comps):
                if wp.carbon.read(part, span, cell, c) != encode(b, x, y, z, c):
                    wp.atomic_add(err_count, 0, 1)
                else:
                    wp.atomic_add(ok_count, 0, 1)

        span = block.get_span()
        part = field.get_partition()
        dim = (int(span.iter_block_count), int(span.mem_x), int(span.mem_y), int(span.mem_z))
        err = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        ok = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        wp.launch(fill, dim=dim, inputs=[part, span, ncomp], device="cuda:0")
        wp.launch(check, dim=dim, inputs=[part, span, ncomp, err, ok], device="cuda:0")
        wp.synchronize()

        err_count = int(err.numpy()[0])
        ok_count = int(ok.numpy()[0])
        self.assertEqual(err_count, 0, f"block mismatches: err={err_count} ok={ok_count}")
        self.assertGreater(ok_count, 0)

        field.delete()
        block.delete()
        system.delete()

    def test_mgrid_cross_level_components(self) -> None:
        wp, carbon, gate = self._bootstrap()
        ncomp = 2
        ref_factor = 2

        system = carbon.System(gate)
        mgrid = carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=ref_factor)
        field = mgrid.new_field(ncomp, dtype=wp.float32)

        @wp.func
        def encode(level: int, x: int, y: int, z: int, c: int):
            return wp.float32(level * 1000000 + c * 100000 + (x + 100 * y + 10000 * z))

        @wp.kernel
        def fill_level(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            level: int,
            comps: int,
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            gx = wp.carbon.global_x(span, b, x, y, z)
            gy = wp.carbon.global_y(span, b, x, y, z)
            gz = wp.carbon.global_z(span, b, x, y, z)
            cell = wp.carbon.set(span, b, x, y, z)
            for c in range(comps):
                wp.carbon.write(part, span, cell, c, encode(level, gx, gy, gz, c))

        for level in range(mgrid.depth):
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
                inputs=[part, span, level, ncomp],
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
            comps: int,
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
            cell = wp.carbon.set(own_span, b, x, y, z)
            for oz in range(rf):
                for oy in range(rf):
                    for ox in range(rf):
                        if not wp.carbon.child_valid(part, b, x, y, z, ox, oy, oz):
                            wp.atomic_add(err_count, 0, 1)
                            continue
                        gx = wp.carbon.child_global_x(part, b, x, y, z, ox, oy, oz)
                        gy = wp.carbon.child_global_y(part, b, x, y, z, ox, oy, oz)
                        gz = wp.carbon.child_global_z(part, b, x, y, z, ox, oy, oz)
                        for c in range(comps):
                            got = wp.carbon.read_child(part, cell, ox, oy, oz, c)
                            if got != encode(0, gx, gy, gz, c):
                                wp.atomic_add(err_count, 0, 1)
                            else:
                                wp.atomic_add(ok_count, 0, 1)

        @wp.kernel
        def check_parents(
            part: gate.CarbonMgridPartition,
            own_span: gate.CarbonBlockSpan,
            comps: int,
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
            gx = wp.carbon.parent_global_x(part, own_span, b, x, y, z)
            gy = wp.carbon.parent_global_y(part, own_span, b, x, y, z)
            gz = wp.carbon.parent_global_z(part, own_span, b, x, y, z)
            for c in range(comps):
                if wp.carbon.read(part, own_span, cell, c) != encode(1, gx, gy, gz, c):
                    wp.atomic_add(err_count, 0, 1)
                else:
                    wp.atomic_add(ok_count, 0, 1)

        own_span = mgrid.get_span(1)
        wp.launch(
            check_children,
            dim=(
                int(own_span.iter_block_count),
                int(own_span.mem_x),
                int(own_span.mem_y),
                int(own_span.mem_z),
            ),
            inputs=[field.get_cross_partition(1), own_span, ref_factor, ncomp, err, ok],
            device="cuda:0",
        )

        own_span = mgrid.get_span(0)
        wp.launch(
            check_parents,
            dim=(
                int(own_span.iter_block_count),
                int(own_span.mem_x),
                int(own_span.mem_y),
                int(own_span.mem_z),
            ),
            inputs=[field.get_cross_partition(0), own_span, ncomp, err, ok],
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
