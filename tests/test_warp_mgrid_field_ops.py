#!/usr/bin/env python3
"""MgridField per-level field ops (C.4): fill_run / zero_run / copy_from_run.

Validates the per-level device field operations mirror the dense API:
- fill_run(level, value) writes ``value`` into every active cell at that level,
- copy_from_run(level, other) duplicates a level between two fields,
- zero_run(level) clears a level,
across all visible devices (each per-device slab checked on its own device).
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpMgridFieldOps(unittest.TestCase):
    def test_field_ops(self) -> None:
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
        n_dev = gate.lib.carbonpy_device_count()
        if n_dev <= 0:
            self.skipTest("no CUDA devices visible")
        carbon.install_neon_aliases()

        system = carbon.System(gate)
        mgrid = carbon.Mgrid(system, 16, 16, 32, depth=2, ref_factor=2)
        a = mgrid.new_field(2, dtype=wp.float32)
        b = mgrid.new_field(2, dtype=wp.float32)

        self.assertEqual(a.num_levels, mgrid.depth)

        @wp.kernel
        def count_equal(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            value: wp.float32,
            ncomp: int,
            ok: wp.array(dtype=wp.int32),
            bad: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            blk = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, blk, x, y, z):
                return
            cell = wp.carbon.set(span, blk, x, y, z)
            for c in range(ncomp):
                got = wp.carbon.read(part, span, cell, c)
                if got == value:
                    wp.atomic_add(ok, 0, 1)
                else:
                    wp.atomic_add(bad, 0, 1)

        def total_active(field, level, value) -> tuple:
            tot_ok = 0
            tot_bad = 0
            for gpu in range(n_dev):
                span = mgrid.get_span(level, gpu, 0)
                part = field.get_partition(level, gpu, 0)
                dim = (
                    int(span.iter_block_count),
                    int(span.mem_x),
                    int(span.mem_y),
                    int(span.mem_z),
                )
                if dim[0] <= 0:
                    continue
                device = wp.get_device(f"cuda:{gpu}")
                ok = wp.zeros(1, dtype=wp.int32, device=device)
                bad = wp.zeros(1, dtype=wp.int32, device=device)
                wp.launch(
                    count_equal,
                    dim=dim,
                    inputs=[part, span, wp.float32(value), int(field.components), ok, bad],
                    device=device,
                )
                wp.synchronize_device(device)
                tot_ok += int(ok.numpy()[0])
                tot_bad += int(bad.numpy()[0])
            return tot_ok, tot_bad

        import numpy as np

        for level in range(mgrid.depth):
            a.fill_run(level, 7.0)
            ok, bad = total_active(a, level, 7.0)
            self.assertEqual(bad, 0, f"fill_run left {bad} mismatched cells at level {level}")
            self.assertGreater(ok, 0, f"fill_run touched no cells at level {level}")

            arr = a.to_numpy(level)
            gx, gy, gz = mgrid.level_extent(level)
            self.assertEqual(arr.shape, (a.components, gx, gy, gz))
            # Every active-cell/component slot counted by ok must read back as 7.0.
            self.assertEqual(int(np.count_nonzero(arr == np.float32(7.0))), ok)

            b.copy_from_run(level, a)
            ok2, bad2 = total_active(b, level, 7.0)
            self.assertEqual(bad2, 0, f"copy_from_run mismatch at level {level}")
            self.assertEqual(ok2, ok, f"copy_from_run active-cell count differs at level {level}")

            b.zero_run(level)
            ok3, _ = total_active(b, level, 0.0)
            self.assertEqual(ok3, ok, f"zero_run active-cell count differs at level {level}")

        a.delete()
        b.delete()
        mgrid.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
