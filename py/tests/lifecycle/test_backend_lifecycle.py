"""Sequential neon.Backend lifecycles must leave Warp usable.

Before CudaDriver switched to the device primary context, destroying a Backend
called cuCtxDestroy on a private context that Warp had registered as cuda:0.N;
the next wp.synchronize() hit CUDA error 709 or crashed.
"""

import gc
import unittest

import numpy as np

from neon_testing import NeonTestCase, grids, non_primary_cuda_devices, operators, require_gpu, run_container

import warp as wp
import neon

CYCLES = 4


def _new_backend():
    return neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=[0])


def _mres_cycle(n=32):
    """Backend -> mGrid -> field -> container -> run, then tear down newest first."""
    bk = _new_backend()
    mask0 = np.zeros((n, n, n), dtype=np.int32)
    h = max(1, n // 4)
    mask0[:h, :h, :h] = 1
    mask1 = np.ones((n // 2,) * 3, dtype=np.int32)
    grid = grids.make_mgrid(bk, n, [mask0, mask1])
    field = grid.new_field(cardinality=1, dtype=wp.int32, memory_type=neon.MemoryType.host_device())
    field.zero_run(1, 0)
    op = operators.mres_add_constant(field, level=1, value=1)
    run_container(op)
    field.update_host(0)
    wp.synchronize()
    probe = grids.active_voxels(grid, 1, n)[0]
    value = field.read(1, probe, 0)
    for obj in (op, field, grid, bk):
        obj.cleanup()
    gc.collect()
    wp.synchronize()
    return value


def _grid_cycle(grid_type):
    dim = neon.Index_3d(4, 4, 4)
    with _new_backend() as bk:
        grid = grid_type(bk, dim)
        field = grid.new_field(cardinality=1, dtype=wp.int32)
        with operators.add_coord_sum(field) as op:
            run_container(op)
        field.update_host(0)
        wp.synchronize()
        value = field.read(idx=neon.Index_3d(1, 2, 3), cardinality=0)
        field.cleanup()
        grid.cleanup()
    wp.synchronize()
    return value


@require_gpu
class TestBackendLifecycle(NeonTestCase):
    def assertOnlyPrimaryContexts(self, msg):
        self.assertEqual(non_primary_cuda_devices(), [], msg)

    def test_backend_without_work(self):
        for i in range(CYCLES):
            _new_backend().cleanup()
            wp.synchronize()
            self.assertOnlyPrimaryContexts(f"cycle {i}")

    def test_repeated_mres_lifecycles(self):
        for i in range(CYCLES):
            with self.subTest(cycle=i):
                self.assertEqual(_mres_cycle(), 1)
                self.assertOnlyPrimaryContexts(f"cycle {i} left a non-primary Warp device")

    def test_dense_and_block_lifecycles_interleaved(self):
        for i in range(CYCLES):
            for grid_type in (neon.dense.dGrid, neon.block.bGrid):
                with self.subTest(cycle=i, grid=grid_type.__name__):
                    # The field starts zeroed, so one add yields the coordinate sum.
                    self.assertEqual(_grid_cycle(grid_type), 6)
            self.assertOnlyPrimaryContexts(f"cycle {i}")

    def test_cleanup_is_idempotent(self):
        bk = _new_backend()
        bk.cleanup()
        self.assertFalse(bk.backend_handle.value)
        self.assertFalse(bk.cuda_driver_handle.value)
        bk.cleanup()
        del bk
        gc.collect()
        wp.synchronize()

    def test_context_manager_releases_on_exception(self):
        with self.assertRaises(ZeroDivisionError):
            with _new_backend() as bk:
                1 / 0
        self.assertFalse(bk.backend_handle.value)
        wp.synchronize()
        self.assertOnlyPrimaryContexts("backend released by __exit__")

    def test_warp_usable_after_backends(self):
        _mres_cycle()
        a = wp.array(np.arange(16, dtype=np.float32), device="cuda:0")
        wp.synchronize()
        np.testing.assert_array_equal(a.numpy(), np.arange(16, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
