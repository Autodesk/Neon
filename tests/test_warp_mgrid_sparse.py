#!/usr/bin/env python3
"""Sparse mGrid construction (C.5): per-level masks + origins.

Validates carbonpy_mgrid_new_sparse (the Neon pyNeon-parity
mGrid(sparsity_pattern_list, sparsity_pattern_origins) interface):
- full-coverage masks reproduce the all-active grid's active-cell count,
- a restricted level-0 mask yields exactly the masked active cells (Carbon's
  MultiResBlock does no strong-balance / overlap-cull, so the active set equals
  the masked voxels within retained bricks),
- masked-in cells read back the fill value; masked-out cells read back zero.
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpMgridSparse(unittest.TestCase):
    def test_sparse_construction(self) -> None:
        root = _repo_root()
        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp  # noqa: F401
        except ImportError:
            self.skipTest("warp module not installed")

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        import numpy as np

        import carbon  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")
        carbon.install_neon_aliases()

        system = carbon.System(gate)
        # n must be a multiple of MemZ*R = 16 so the coarsest level has >= 1 brick row per slab
        # on a multi-GPU box; n=32 gives a 16^3 coarsest (2 bricks in z).
        n = 32

        def active_count(field, level) -> int:
            arr = field.to_numpy(level)
            return int(np.count_nonzero(arr == np.float32(7.0)))

        # --- Reference: all-active grid (default ctor). ---
        ref = carbon.Mgrid(system, n, n, n, depth=2, ref_factor=2)
        ref_field = ref.new_field(1, dtype=wp.float32)
        ref_field.fill_run(0, 7.0)
        full_l0 = active_count(ref_field, 0)
        self.assertEqual(full_l0, n * n * n)  # 4096.
        ref_field.delete()
        ref.delete()

        # --- Full-coverage sparse masks reproduce the all-active grid. ---
        masks_full = [
            np.ones((n, n, n), dtype=np.int32),
            np.ones((n // 2, n // 2, n // 2), dtype=np.int32),
        ]
        sfull = carbon.Mgrid(
            system, n, n, n, ref_factor=2, sparsity_pattern_list=masks_full
        )
        self.assertEqual(sfull.depth, 2)
        sfull_field = sfull.new_field(1, dtype=wp.float32)
        sfull_field.fill_run(0, 7.0)
        self.assertEqual(active_count(sfull_field, 0), full_l0)
        sfull_field.delete()
        sfull.delete()

        # --- Restricted level-0 mask: activate only the x in [0, n/2) brick columns. ---
        half = n // 2
        m0 = np.zeros((n, n, n), dtype=np.int32)
        m0[0:half, :, :] = 1  # brick-aligned half-domain -> half*n*n cells.
        masks = [m0, np.ones((n // 2, n // 2, n // 2), dtype=np.int32)]
        sparse = carbon.Mgrid(
            system, n, n, n, ref_factor=2, sparsity_pattern_list=masks
        )
        sp_field = sparse.new_field(1, dtype=wp.float32)
        sp_field.fill_run(0, 7.0)

        got = active_count(sp_field, 0)
        self.assertEqual(got, half * n * n)
        self.assertLess(got, full_l0)  # genuinely sparse.

        arr = sp_field.to_numpy(0)  # (1, n, n, n)
        self.assertEqual(arr[0, 0, 0, 0], np.float32(7.0))  # masked-in -> active.
        self.assertEqual(arr[0, n - 1, 0, 0], np.float32(0.0))  # masked-out -> inactive.

        sp_field.delete()
        sparse.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
