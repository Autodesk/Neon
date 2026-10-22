#!/usr/bin/env python3
"""Multires LBM builtins on the fat mgrid cross partition (C.3).

Validates the wp.neon_* multires surface against a 2-level all-active octree:
- has_child / has_parent level semantics (finest has parent + no child; coarsest the reverse),
- lbm_read_coarser_ngh with a zero pull direction reduces to the direct-parent read,
- has_finer_ngh + mres_lbm_store_op codegen/run (atomic explosion-store path compiles).
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpMgridMultires(unittest.TestCase):
    def test_multires_builtins(self) -> None:
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
        carbon.install_neon_aliases()

        system = carbon.System(gate)
        mgrid = carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=2)
        field = mgrid.new_field(1, dtype=wp.float32)

        @wp.func
        def encode(gx: int, gy: int, gz: int):
            return wp.float32(gx + 100 * gy + 10000 * gz)

        # Fill the coarse level (1) own storage with encode(global).
        @wp.kernel
        def fill(part: gate.CarbonMgridPartition, span: gate.CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(part, iter_idx, x, y, z)
            g = wp.carbon.global_idx(part, cell)
            wp.carbon.write(
                part, cell, 0, encode(wp.carbon.get_x(g), wp.carbon.get_y(g), wp.carbon.get_z(g))
            )

        # Coarsest level: every active cell is refined (has_child) and has no parent.
        @wp.kernel
        def check_coarse(
            part: gate.CarbonMgridPartition,
            span: gate.CarbonBlockSpan,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(part, iter_idx, x, y, z)
            if not wp.neon_has_child(part, cell):
                wp.atomic_add(err_count, 0, 1)
                return
            if wp.neon_has_parent(part, cell):
                wp.atomic_add(err_count, 0, 1)
                return
            wp.atomic_add(ok_count, 0, 1)

        # Finest level: every active cell has a parent, no child; the zero-direction
        # coarser-read equals the direct parent read; the store path compiles/runs.
        @wp.kernel
        def check_fine(
            part: gate.CarbonMgridPartition,
            span: gate.CarbonBlockSpan,
            err_count: wp.array(dtype=wp.int32),
            ok_count: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(part, iter_idx, x, y, z)
            if wp.neon_has_child(part, cell):
                wp.atomic_add(err_count, 0, 1)
                return
            if not wp.neon_has_parent(part, cell):
                wp.atomic_add(err_count, 0, 1)
                return
            parent_val = wp.carbon.read(part, span, cell, 0)  # 3-arg = parent read
            zero = wp.neon_ngh_idx(wp.int8(0), wp.int8(0), wp.int8(0))
            valid = bool(False)
            ex = wp.neon_lbm_read_coarser_ngh(part, cell, zero, 0, wp.float32(-1.0), valid)
            if (not valid) or ex != parent_val:
                wp.atomic_add(err_count, 0, 1)
            else:
                wp.atomic_add(ok_count, 0, 1)
            # has_finer_ngh: exercise the path (interior +x neighbour is refined).
            px = wp.neon_ngh_idx(wp.int8(1), wp.int8(0), wp.int8(0))
            if wp.neon_has_finer_ngh(part, cell, px):
                wp.atomic_add(ok_count, 0, 1)
            # Store path (atomic explosion-store) — smoke: must compile and run.
            wp.neon_mres_lbm_store_op(part, cell, 0, px, wp.float32(0.0))

        span1 = mgrid.get_span(1)
        part1 = field.get_cross_partition(1)
        wp.launch(
            fill,
            dim=(int(span1.iter_block_count), int(span1.mem_x), int(span1.mem_y), int(span1.mem_z)),
            inputs=[part1, span1],
            device="cuda:0",
        )
        wp.synchronize()

        err = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        ok = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        wp.launch(
            check_coarse,
            dim=(int(span1.iter_block_count), int(span1.mem_x), int(span1.mem_y), int(span1.mem_z)),
            inputs=[part1, span1, err, ok],
            device="cuda:0",
        )

        span0 = mgrid.get_span(0)
        part0 = field.get_cross_partition(0)
        wp.launch(
            check_fine,
            dim=(int(span0.iter_block_count), int(span0.mem_x), int(span0.mem_y), int(span0.mem_z)),
            inputs=[part0, span0, err, ok],
            device="cuda:0",
        )
        wp.synchronize()

        err_count = int(err.numpy()[0])
        ok_count = int(ok.numpy()[0])
        self.assertEqual(err_count, 0, f"multires builtin mismatches: err={err_count} ok={ok_count}")
        self.assertGreater(ok_count, 0)

        field.delete()
        mgrid.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
