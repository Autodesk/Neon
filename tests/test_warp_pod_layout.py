#!/usr/bin/env python3
"""Verify ctypes POD layouts match the libcarbonpy C-ABI introspection exports."""

import ctypes
import os
import sys
import unittest


def _field_offset(struct_cls: type, field_name: str) -> int:
    return getattr(struct_cls, field_name).offset


def _load_lib():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_dir = os.path.join(root, "py", "carbon", "lib", f"linux_{os.uname().machine}")
    lib_path = os.path.join(lib_dir, "libcarbonpy.so")
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(lib_path)
    lib = ctypes.CDLL(lib_path)
    for name in (
        "carbonpy_sizeof_dense_span",
        "carbonpy_offsetof_dense_span_iter_x",
        "carbonpy_offsetof_dense_span_iter_y",
        "carbonpy_offsetof_dense_span_iter_z",
        "carbonpy_offsetof_dense_span_logic_x",
        "carbonpy_offsetof_dense_span_logic_y",
        "carbonpy_offsetof_dense_span_logic_z",
        "carbonpy_offsetof_dense_span_pitch_xy",
        "carbonpy_offsetof_dense_span_z_offset",
        "carbonpy_offsetof_dense_span_halo_x",
        "carbonpy_offsetof_dense_span_halo_y",
        "carbonpy_offsetof_dense_span_halo_z",
        "carbonpy_offsetof_dense_span_global_x",
        "carbonpy_offsetof_dense_span_global_y",
        "carbonpy_offsetof_dense_span_global_z",
        "carbonpy_offsetof_dense_span_boundary_mode",
        "carbonpy_offsetof_dense_span_data_view",
        "carbonpy_sizeof_block_span",
        "carbonpy_offsetof_block_span_bitmask_ptr",
        "carbonpy_offsetof_block_span_connectivity_ptr",
        "carbonpy_offsetof_block_span_origins_ptr",
        "carbonpy_offsetof_block_span_block_count",
        "carbonpy_offsetof_block_span_n_internal",
        "carbonpy_offsetof_block_span_iter_block_count",
        "carbonpy_offsetof_block_span_mem_x",
        "carbonpy_offsetof_block_span_mem_y",
        "carbonpy_offsetof_block_span_mem_z",
        "carbonpy_sizeof_mgrid_partition",
        "carbonpy_offsetof_mgrid_partition_data_ptr",
        "carbonpy_offsetof_mgrid_partition_ref_factor",
        "carbonpy_offsetof_mgrid_partition_components",
        "carbonpy_offsetof_mgrid_partition_own_span",
        "carbonpy_offsetof_mgrid_partition_parent_data_ptr",
        "carbonpy_offsetof_mgrid_partition_parent_block_id_ptr",
        "carbonpy_offsetof_mgrid_partition_parent_span",
        "carbonpy_offsetof_mgrid_partition_child_data_ptr",
        "carbonpy_offsetof_mgrid_partition_child_block_id_ptr",
        "carbonpy_offsetof_mgrid_partition_child_span",
    ):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = []
    return lib


class TestWarpPodLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
        try:
            cls.lib = _load_lib()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        from carbon.warp_types import CarbonBlockSpan, CarbonDenseSpan, CarbonMgridPartitionF32

        cls.CarbonDenseSpan = CarbonDenseSpan
        cls.CarbonBlockSpan = CarbonBlockSpan
        cls.CarbonMgridPartitionF32 = CarbonMgridPartitionF32

    def test_dense_span_layout(self) -> None:
        span = self.CarbonDenseSpan
        self.assertEqual(ctypes.sizeof(span), self.lib.carbonpy_sizeof_dense_span())
        pairs = (
            ("iter_x", self.lib.carbonpy_offsetof_dense_span_iter_x),
            ("iter_y", self.lib.carbonpy_offsetof_dense_span_iter_y),
            ("iter_z", self.lib.carbonpy_offsetof_dense_span_iter_z),
            ("logic_x", self.lib.carbonpy_offsetof_dense_span_logic_x),
            ("logic_y", self.lib.carbonpy_offsetof_dense_span_logic_y),
            ("logic_z", self.lib.carbonpy_offsetof_dense_span_logic_z),
            ("pitch_xy", self.lib.carbonpy_offsetof_dense_span_pitch_xy),
            ("z_offset", self.lib.carbonpy_offsetof_dense_span_z_offset),
            ("halo_x", self.lib.carbonpy_offsetof_dense_span_halo_x),
            ("halo_y", self.lib.carbonpy_offsetof_dense_span_halo_y),
            ("halo_z", self.lib.carbonpy_offsetof_dense_span_halo_z),
            ("global_x", self.lib.carbonpy_offsetof_dense_span_global_x),
            ("global_y", self.lib.carbonpy_offsetof_dense_span_global_y),
            ("global_z", self.lib.carbonpy_offsetof_dense_span_global_z),
            ("boundary_mode", self.lib.carbonpy_offsetof_dense_span_boundary_mode),
            ("data_view", self.lib.carbonpy_offsetof_dense_span_data_view),
        )
        for field, c_fn in pairs:
            self.assertEqual(_field_offset(span, field), c_fn(), field)

    def test_block_span_layout(self) -> None:
        span = self.CarbonBlockSpan
        self.assertEqual(ctypes.sizeof(span), self.lib.carbonpy_sizeof_block_span())
        pairs = (
            ("bitmask_ptr", self.lib.carbonpy_offsetof_block_span_bitmask_ptr),
            ("connectivity_ptr", self.lib.carbonpy_offsetof_block_span_connectivity_ptr),
            ("origins_ptr", self.lib.carbonpy_offsetof_block_span_origins_ptr),
            ("block_count", self.lib.carbonpy_offsetof_block_span_block_count),
            ("n_internal", self.lib.carbonpy_offsetof_block_span_n_internal),
            ("iter_block_count", self.lib.carbonpy_offsetof_block_span_iter_block_count),
            ("mem_x", self.lib.carbonpy_offsetof_block_span_mem_x),
            ("mem_y", self.lib.carbonpy_offsetof_block_span_mem_y),
            ("mem_z", self.lib.carbonpy_offsetof_block_span_mem_z),
        )
        for field, c_fn in pairs:
            self.assertEqual(_field_offset(span, field), c_fn(), field)

    def test_mgrid_partition_layout(self) -> None:
        part = self.CarbonMgridPartitionF32
        self.assertEqual(ctypes.sizeof(part), self.lib.carbonpy_sizeof_mgrid_partition())
        pairs = (
            ("data_ptr", self.lib.carbonpy_offsetof_mgrid_partition_data_ptr),
            ("ref_factor", self.lib.carbonpy_offsetof_mgrid_partition_ref_factor),
            ("components", self.lib.carbonpy_offsetof_mgrid_partition_components),
            ("own_span", self.lib.carbonpy_offsetof_mgrid_partition_own_span),
            ("parent_data_ptr", self.lib.carbonpy_offsetof_mgrid_partition_parent_data_ptr),
            ("parent_block_id_ptr", self.lib.carbonpy_offsetof_mgrid_partition_parent_block_id_ptr),
            ("parent_span", self.lib.carbonpy_offsetof_mgrid_partition_parent_span),
            ("child_data_ptr", self.lib.carbonpy_offsetof_mgrid_partition_child_data_ptr),
            ("child_block_id_ptr", self.lib.carbonpy_offsetof_mgrid_partition_child_block_id_ptr),
            ("child_span", self.lib.carbonpy_offsetof_mgrid_partition_child_span),
        )
        for field, c_fn in pairs:
            self.assertEqual(_field_offset(part, field), c_fn(), field)


if __name__ == "__main__":
    unittest.main()
