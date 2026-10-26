"""Round-trip tests for mGrid topology + field checkpoint I/O."""

import json
import os
import struct
import tempfile
import unittest

import numpy as np

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon
from neon.multires.mField import FieldError, mField

DIM = 16
STENCIL = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]


def _make_grid(backend, fine_extent):
    """Two-level grid: a fine box [0, fine_extent)^2 x [0, 4) on level 0, everything on level 1."""
    level0 = np.zeros((DIM, DIM, DIM), dtype=np.int32)
    level0[:fine_extent, :fine_extent, :4] = 1
    level1 = np.ones((DIM // 2, DIM // 2, DIM // 2), dtype=np.int32)
    return neon.mGrid(
        backend,
        neon.Index_3d(DIM, DIM, DIM),
        sparsity_pattern_list=[np.ascontiguousarray(level0), np.ascontiguousarray(level1)],
        sparsity_pattern_origins=[neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
        stencil=STENCIL,
    )


def _active_voxels(grid, level):
    stride = 1 << level
    voxels = []
    for z in range(0, DIM, stride):
        for y in range(0, DIM, stride):
            for x in range(0, DIM, stride):
                idx = neon.Index_3d(x, y, z)
                if grid.is_inside_domain(level, idx):
                    voxels.append((x, y, z))
    return voxels


def _value(level, x, y, z, c):
    return float(x + 10 * y + 100 * z + 1000 * level + 0.5 * c)


class TestMresCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.backend = neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=[0])

    def _fill(self, grid, field, cardinality):
        for level in range(grid.num_levels):
            for (x, y, z) in _active_voxels(grid, level):
                for c in range(cardinality):
                    res = field.write(level, neon.Index_3d(x, y, z), c, _value(level, x, y, z, c))
                    self.assertEqual(res, 0)
        field.update_device(0)
        wp.synchronize()

    def _check_values(self, grid, field, cardinality, scale=1.0):
        for level in range(grid.num_levels):
            voxels = _active_voxels(grid, level)
            self.assertGreater(len(voxels), 0)
            for (x, y, z) in voxels:
                for c in range(cardinality):
                    got = field.read(level, neon.Index_3d(x, y, z), c)
                    self.assertEqual(got, scale * _value(level, x, y, z, c),
                                     f"level={level} idx=({x},{y},{z}) c={c}")

    def _roundtrip(self, cardinality, file_format):
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(cardinality, wp.float32, neon.MemoryType.host_device())
        self._fill(grid, field, cardinality)

        with tempfile.TemporaryDirectory() as tmp:
            topo = os.path.join(tmp, "grid.ngrid")
            base = os.path.join(tmp, "rho")
            grid.save_topology(topo)
            field.save(base, name="rho", file_format=file_format)

            with open(base + ".json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["name"], "rho")
            self.assertEqual(meta["dtype"], "float32")
            self.assertEqual(meta["cardinality"], cardinality)
            self.assertEqual(meta["format"], file_format)
            self.assertTrue(os.path.exists(base + "." + file_format))
            for other in set(mField.CHECKPOINT_FORMATS) - {file_format}:
                self.assertFalse(os.path.exists(base + "." + other))
            if file_format == "bin":
                self.assertEqual(os.path.getsize(base + ".bin"), meta["num_elements"] * 4)

            grid2 = neon.mGrid.from_topology(self.backend, topo)
            self.assertEqual(grid2.num_levels, grid.num_levels)
            for level in range(grid.num_levels):
                self.assertEqual(_active_voxels(grid2, level), _active_voxels(grid, level))
            field2 = grid2.load_field(base)
            self.assertEqual(field2.cardinality, cardinality)
            self._check_values(grid, field2, cardinality)

    def test_float32_cardinality1_roundtrip(self):
        for file_format in mField.CHECKPOINT_FORMATS:
            with self.subTest(file_format=file_format):
                self._roundtrip(cardinality=1, file_format=file_format)

    def test_float32_cardinality3_roundtrip(self):
        for file_format in mField.CHECKPOINT_FORMATS:
            with self.subTest(file_format=file_format):
                self._roundtrip(cardinality=3, file_format=file_format)

    def test_numpy_files_match_bin(self):
        cardinality = 3
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(cardinality, wp.float32, neon.MemoryType.host_device())
        self._fill(grid, field, cardinality)

        with tempfile.TemporaryDirectory() as tmp:
            for file_format in mField.CHECKPOINT_FORMATS:
                field.save(os.path.join(tmp, file_format), name="v", file_format=file_format)
            with open(os.path.join(tmp, "npz.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            z, y, x = meta["block_shape_zyx"]
            blocks = meta["num_blocks_per_level"]
            self.assertEqual(len(blocks), grid.num_levels)
            self.assertEqual(meta["layout"], "block, z, y, x, component")

            raw = np.fromfile(os.path.join(tmp, "bin.bin"), dtype="<f4")
            self.assertEqual(raw.size, sum(blocks) * z * y * x * cardinality)

            npy = np.load(os.path.join(tmp, "npy.npy"), allow_pickle=False)
            self.assertEqual(npy.dtype, np.float32)
            self.assertEqual(npy.shape, (sum(blocks), z, y, x, cardinality))
            np.testing.assert_array_equal(npy.reshape(-1), raw)

            with np.load(os.path.join(tmp, "npz.npz"), allow_pickle=False) as npz:
                self.assertEqual(sorted(npz.files), [f"level_{l}" for l in range(grid.num_levels)])
                levels = [npz[f"level_{l}"] for l in range(grid.num_levels)]
            for l, arr in enumerate(levels):
                self.assertEqual(arr.shape, (blocks[l], z, y, x, cardinality))
            np.testing.assert_array_equal(np.concatenate(levels), npy)

    def test_load_files_written_by_numpy(self):
        cardinality = 3
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(cardinality, wp.float32, neon.MemoryType.host_device())
        self._fill(grid, field, cardinality)

        with tempfile.TemporaryDirectory() as tmp:
            npy_base = os.path.join(tmp, "a")
            npz_base = os.path.join(tmp, "b")
            field.save(npy_base, name="v", file_format="npy")
            field.save(npz_base, name="v", file_format="npz")

            np.save(npy_base + ".npy", 2.0 * np.load(npy_base + ".npy"))
            with np.load(npz_base + ".npz") as npz:
                doubled = {k: 2.0 * npz[k] for k in npz.files}
            np.savez(npz_base + ".npz", **doubled)

            for base in (npy_base, npz_base):
                with self.subTest(base=os.path.basename(base)):
                    restored = grid.new_field(cardinality, wp.float32, neon.MemoryType.host_device())
                    restored.load(base)
                    self._check_values(grid, restored, cardinality, scale=2.0)

            np.savez_compressed(npz_base + ".npz", **doubled)
            restored = grid.new_field(cardinality, wp.float32, neon.MemoryType.host_device())
            with self.assertRaises(FieldError):
                restored.load(npz_base)

            np.save(npy_base + ".npy", np.load(npy_base + ".npy").astype(np.float64))
            with self.assertRaises(FieldError):
                restored.load(npy_base)

    def test_host_write_lands_at_coordinate(self):
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(1, wp.float32, neon.MemoryType.host_device())

        def encode(x, y, z):
            return float(1 + x + 100 * y + 10000 * z)

        for level in range(grid.num_levels):
            for (x, y, z) in _active_voxels(grid, level):
                self.assertEqual(field.write(level, neon.Index_3d(x, y, z), 0, encode(x, y, z)), 0)
        field.update_device(0)
        wp.synchronize()

        with tempfile.TemporaryDirectory() as tmp:
            topo = os.path.join(tmp, "grid.ngrid")
            grid.save_topology(topo)
            field.save(os.path.join(tmp, "f"), file_format="npz")
            with open(topo, "rb") as f:
                raw = f.read()
            header_len = struct.unpack("<Q", raw[8:16])[0]
            header = json.loads(raw[16:16 + header_len])
            with np.load(os.path.join(tmp, "f.npz"), allow_pickle=False) as npz:
                levels = [npz[f"level_{l}"] for l in range(grid.num_levels)]

        for level, arr in enumerate(levels):
            meta = header["levels"][level]
            section = meta["block_origins"]
            start = 16 + header_len + section["offset"]
            origins = np.frombuffer(raw[start:start + section["bytes"]], dtype="<i4").reshape(section["shape"])
            spacing = meta["voxel_spacing"]
            slot_value = {}
            for b, origin in enumerate(origins):
                for zz in range(arr.shape[1]):
                    for yy in range(arr.shape[2]):
                        for xx in range(arr.shape[3]):
                            coord = (int(origin[0]) + xx * spacing,
                                     int(origin[1]) + yy * spacing,
                                     int(origin[2]) + zz * spacing)
                            slot_value[coord] = float(arr[b, zz, yy, xx, 0])
            for (x, y, z) in _active_voxels(grid, level):
                self.assertEqual(slot_value.get((x, y, z)), encode(x, y, z), f"level={level} idx=({x},{y},{z})")

    def test_invalid_format_rejected(self):
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(1, wp.float32, neon.MemoryType.host_device())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                field.save(os.path.join(tmp, "rho"), file_format="hdf5")

    def test_topology_id_mismatch_rejected(self):
        grid = _make_grid(self.backend, fine_extent=8)
        field = grid.new_field(1, wp.float32, neon.MemoryType.host_device())
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "rho")
            field.save(base, name="rho")

            other = _make_grid(self.backend, fine_extent=12)
            other_field = other.new_field(1, wp.float32, neon.MemoryType.host_device())
            with self.assertRaises(FieldError):
                other_field.load(base)


if __name__ == "__main__":
    unittest.main()
