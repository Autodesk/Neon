"""mField.export_vti writes non-empty files (always into a temporary directory)."""

import os
import tempfile
import unittest

from neon_testing import NeonTestCase, grids, operators, require_gpu

import warp as wp
import neon


@require_gpu
class TestMresExport(NeonTestCase):
    def setUp(self):
        super().setUp()
        self.grid, _, _ = grids.two_level_mgrid(self.backend())
        self.track(self.grid)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _export(self, field, name):
        field.update_host(0)
        wp.synchronize()
        field.export_vti(os.path.join(self.tmp.name, name), "field")
        written = [f for f in os.listdir(self.tmp.name) if f.startswith(name)]
        self.assertTrue(written, f"export_vti wrote nothing for {name}")
        for f in written:
            self.assertGreater(os.path.getsize(os.path.join(self.tmp.name, f)), 0, f)

    def test_export_int32_scalar(self):
        field = self.track(self.grid.new_field(cardinality=1, dtype=wp.int32,
                                               memory_type=neon.MemoryType.host_device()))
        for level in range(2):
            self.run_op(operators.mres_set_coord_sum(field, level=level))
        self._export(field, "int_scalar")

    def test_export_float32_vector(self):
        field = self.track(self.grid.new_field(cardinality=3, dtype=wp.float32,
                                               memory_type=neon.MemoryType.host_device()))
        for level in range(2):
            field.fill_run(level, 1.5, 0)
        wp.synchronize()
        self._export(field, "float_vector")


if __name__ == "__main__":
    unittest.main()
