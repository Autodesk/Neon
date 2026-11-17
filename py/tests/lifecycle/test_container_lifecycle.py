"""Explicit release of Containers and Skeletons, and use-after-release errors."""

import unittest

from neon_testing import NeonTestCase, grids, operators, require_gpu, run_container

import warp as wp
import neon

DIM = neon.Index_3d(4, 4, 4)
REPETITIONS = 20


@require_gpu
class TestContainerLifecycle(NeonTestCase):
    def setUp(self):
        super().setUp()
        self.bk = self.backend()
        self.grid = self.track(neon.dense.dGrid(self.bk, DIM))
        self.field = self.track(self.grid.new_field(cardinality=1, dtype=wp.int32))

    def _read_back(self):
        self.field.update_host(0)
        wp.synchronize()
        return {(i.x, i.y, i.z): self.field.read(idx=i, cardinality=0) for i in grids.iter_dim(DIM)}

    def _expected(self, factor):
        return {(i.x, i.y, i.z): factor * grids.coord_sum(i) for i in grids.iter_dim(DIM)}

    def test_cleanup_is_idempotent(self):
        op = operators.add_coord_sum(self.field)
        self.assertTrue(op.container_handle.value)
        op.cleanup()
        self.assertFalse(op.container_handle.value)
        op.cleanup()

    def test_run_after_cleanup_raises(self):
        op = operators.add_coord_sum(self.field)
        op.cleanup()
        with self.assertRaisesRegex(RuntimeError, "after cleanup"):
            run_container(op)

    def test_context_manager(self):
        with operators.add_coord_sum(self.field) as op:
            run_container(op)
        self.assertFalse(op.container_handle.value)
        self.assertEqual(self._read_back(), self._expected(1))

    def test_repeated_create_run_cleanup(self):
        for _ in range(REPETITIONS):
            op = operators.add_coord_sum(self.field)
            run_container(op)
            op.cleanup()
        self.assertEqual(self._read_back(), self._expected(REPETITIONS))

    def test_skeleton_rejects_released_container(self):
        op = operators.add_coord_sum(self.field)
        op.cleanup()
        skeleton = self.track(neon.Skeleton(backend=self.bk))
        with self.assertRaises(RuntimeError):
            skeleton.sequence("NeonTestReleased", [op])

    def test_skeleton_run_after_cleanup_raises(self):
        op = self.track(operators.add_coord_sum(self.field))
        skeleton = neon.Skeleton(backend=self.bk)
        skeleton.sequence("NeonTestSkeletonCleanup", [op])
        skeleton.cleanup()
        skeleton.cleanup()
        with self.assertRaisesRegex(RuntimeError, "after cleanup"):
            skeleton.run()

    def test_skeleton_context_manager(self):
        op = self.track(operators.add_coord_sum(self.field))
        with neon.Skeleton(backend=self.bk) as skeleton:
            skeleton.sequence("NeonTestSkeletonWith", [op])
            skeleton.run()
            wp.synchronize()
        with self.assertRaises(RuntimeError):
            skeleton.run()
        self.assertEqual(self._read_back(), self._expected(1))

    def test_container_outlives_skeleton(self):
        op = self.track(operators.add_coord_sum(self.field))
        with neon.Skeleton(backend=self.bk) as skeleton:
            skeleton.sequence("NeonTestSkeletonFirst", [op])
            skeleton.run()
        run_container(op)
        self.assertEqual(self._read_back(), self._expected(2))


if __name__ == "__main__":
    unittest.main()
