"""Skeletons over a four-level mGrid, with map and stencil_up loaders."""

import unittest

from neon_testing import NeonTestCase, export_vti_if_requested, grids, masks, operators, require_gpu, run_container

import warp as wp
import neon

SHELL_WIDTHS = (17, 7, 4)


class _FourLevelCase(NeonTestCase):
    DIM = 64

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bk = cls.class_backend()
        level_masks = masks.nested_shell_levels(cls.DIM, SHELL_WIDTHS)
        cls.num_levels = len(level_masks)
        cls.grid = cls.track_class(grids.make_mgrid(cls.bk, cls.DIM, level_masks))
        cls.active = [grids.active_voxels(cls.grid, l, cls.DIM) for l in range(cls.num_levels)]

    def _field(self):
        field = self.track(self.grid.new_field(cardinality=3, dtype=wp.int32,
                                               memory_type=neon.MemoryType.host_device()))
        for level in range(self.num_levels):
            field.zero_run(level, 0)
        wp.synchronize()
        return field

    def _run_skeleton(self, name, ops, repetitions):
        for op in ops:
            self.track(op)
        skeleton = self.track(neon.Skeleton(backend=self.bk))
        skeleton.sequence(name, ops)
        for _ in range(repetitions):
            skeleton.run()
        wp.synchronize()


@require_gpu
class TestMresSkeleton(_FourLevelCase):
    def setUp(self):
        super().setUp()
        self.field_a = self._field()
        self.field_b = self._field()

    def _mismatches(self, level, expected):
        self.field_a.update_host(0)
        wp.synchronize()
        bad = []
        for v in self.active[level]:
            got = [self.field_a.read(level, v, c) for c in range(3)]
            want = expected(v)
            if got != want:
                bad.append(((v.x, v.y, v.z), got, want))
        return bad[:5]

    def test_every_level_is_populated(self):
        for level in range(self.num_levels):
            self.assertGreater(len(self.active[level]), 0, f"level {level}")

    def test_set_add_copy_check_round_trip(self):
        levels = range(self.num_levels)
        ops = ([operators.mres_write_global_idx(self.field_a, level=l) for l in levels]
               + [operators.mres_add_constant(self.field_a, level=l, value=l) for l in levels]
               + [operators.mres_copy(self.field_a, self.field_b, level=l) for l in levels]
               + [operators.mres_check_global_idx(self.field_b, self.field_a, level=l, offset=l) for l in levels])
        self._run_skeleton("NeonTestMresRoundTrip", ops, repetitions=3)
        export_vti_if_requested(self.field_a, "mres_skeleton_round_trip")

        for level in levels:
            with self.subTest(level=level):
                self.assertEqual(self._mismatches(level, lambda v: [1, 1, 1]), [])

    def test_stencil_up_write(self):
        last = self.num_levels - 1
        ops = [operators.mres_write_global_idx(
                   self.field_a, level=l,
                   operation=neon.Loader.Operation.stencil_up if l < last else neon.Loader.Operation.map)
               for l in range(self.num_levels)]
        self._run_skeleton("NeonTestMresStencilUp", ops, repetitions=3)
        export_vti_if_requested(self.field_a, "mres_skeleton_stencil_up")

        for level in range(self.num_levels):
            with self.subTest(level=level):
                self.assertEqual(self._mismatches(level, lambda v: [v.x, v.y, v.z]), [])


@require_gpu
class TestMresEmptyLevel(_FourLevelCase):
    """At 32^3 the coarsest level is fully covered by finer ones and has no active voxels."""

    DIM = 32

    def test_coarsest_level_is_empty(self):
        self.assertEqual(len(self.active[-1]), 0)
        self.assertGreater(len(self.active[0]), 0)

    def test_container_on_empty_level_is_skipped(self):
        field = self._field()
        run_container(self.track(operators.mres_add_constant(field, level=self.num_levels - 1, value=1)))
        wp.synchronize()

    def test_skeleton_with_empty_level(self):
        field = self._field()
        ops = [operators.mres_add_constant(field, level=l, value=l + 1) for l in range(self.num_levels)]
        self._run_skeleton("NeonTestMresEmptyLevel", ops, repetitions=2)
        field.update_host(0)
        wp.synchronize()
        for level in range(self.num_levels - 1):
            with self.subTest(level=level):
                v = self.active[level][0]
                self.assertEqual(field.read(level, v, 0), 2 * (level + 1))


if __name__ == "__main__":
    unittest.main()
