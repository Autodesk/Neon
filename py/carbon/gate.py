import ctypes
import os
import platform


class Gate:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        plat = platform.system().lower()
        machine = platform.machine().lower()
        if plat.startswith("linux"):
            lib_name = "libcarbonpy.so"
            sub = f"linux_{machine}"
        else:
            raise RuntimeError(f"Unsupported platform for carbonpy: {plat}")

        lib_path = os.path.join(here, "lib", sub, lib_name)
        if not os.path.isfile(lib_path):
            raise RuntimeError(f"libcarbonpy not found at {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self.handle_type = ctypes.c_void_p

        self.lib.carbonpy_init.argtypes = []
        self.lib.carbonpy_init.restype = ctypes.c_int
        self.lib.carbonpy_device_count.argtypes = []
        self.lib.carbonpy_device_count.restype = ctypes.c_int

        self.lib.carbonpy_system_new.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_system_new.restype = ctypes.c_int
        self.lib.carbonpy_system_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_system_delete.restype = ctypes.c_int

        self.lib.carbonpy_dense_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        self.lib.carbonpy_dense_new.restype = ctypes.c_int
        self.lib.carbonpy_dense_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_dense_delete.restype = ctypes.c_int

        self.lib.carbonpy_dense_field_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.carbonpy_dense_field_new.restype = ctypes.c_int
        self.lib.carbonpy_dense_field_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_dense_field_delete.restype = ctypes.c_int

        from .warp_types import (
            CarbonBlockPartition,
            CarbonBlockPartitionF32,
            CarbonBlockPartitionF64,
            CarbonBlockPartitionI32,
            CarbonBlockPartitionI64,
            CarbonBlockSpan,
            CarbonDensePartition,
            CarbonDensePartitionF32,
            CarbonDensePartitionF64,
            CarbonDensePartitionI32,
            CarbonDensePartitionI64,
            CarbonDenseSpan,
            CarbonMgridPartition,
            CarbonMgridPartitionF32,
            CarbonMgridPartitionF64,
            CarbonMgridPartitionI32,
            CarbonMgridPartitionI64,
            _CarbonBlockPartitionLayout,
            _CarbonDensePartitionLayout,
            _CarbonMgridPartitionLayout,
        )

        self.CarbonDenseSpan = CarbonDenseSpan
        self.CarbonDensePartition = CarbonDensePartition
        self.CarbonDensePartitionF32 = CarbonDensePartitionF32
        self.CarbonDensePartitionF64 = CarbonDensePartitionF64
        self.CarbonDensePartitionI32 = CarbonDensePartitionI32
        self.CarbonDensePartitionI64 = CarbonDensePartitionI64

        self.lib.carbonpy_dense_get_span.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CarbonDenseSpan),
        ]
        self.lib.carbonpy_dense_get_span.restype = ctypes.c_int
        self.lib.carbonpy_dense_field_get_partition.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_CarbonDensePartitionLayout),
        ]
        self.lib.carbonpy_dense_field_get_partition.restype = ctypes.c_int

        self.lib.carbonpy_dense_halo_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
        ]
        self.lib.carbonpy_dense_halo_new.restype = ctypes.c_int
        self.lib.carbonpy_dense_halo_run.argtypes = [
            self.handle_type,
            self.handle_type,
        ]
        self.lib.carbonpy_dense_halo_run.restype = ctypes.c_int
        self.lib.carbonpy_dense_halo_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_dense_halo_delete.restype = ctypes.c_int

        self.lib.carbonpy_block_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.carbonpy_block_new.restype = ctypes.c_int
        self.lib.carbonpy_block_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_block_delete.restype = ctypes.c_int
        self.lib.carbonpy_block_field_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.carbonpy_block_field_new.restype = ctypes.c_int
        self.lib.carbonpy_block_field_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_block_field_delete.restype = ctypes.c_int

        self.CarbonBlockSpan = CarbonBlockSpan
        self.CarbonBlockPartition = CarbonBlockPartition
        self.CarbonBlockPartitionF32 = CarbonBlockPartitionF32
        self.CarbonBlockPartitionF64 = CarbonBlockPartitionF64
        self.CarbonBlockPartitionI32 = CarbonBlockPartitionI32
        self.CarbonBlockPartitionI64 = CarbonBlockPartitionI64
        self.CarbonMgridPartition = CarbonMgridPartition
        self.CarbonMgridPartitionF32 = CarbonMgridPartitionF32
        self.CarbonMgridPartitionF64 = CarbonMgridPartitionF64
        self.CarbonMgridPartitionI32 = CarbonMgridPartitionI32
        self.CarbonMgridPartitionI64 = CarbonMgridPartitionI64

        self.lib.carbonpy_block_get_span.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CarbonBlockSpan),
        ]
        self.lib.carbonpy_block_get_span.restype = ctypes.c_int
        self.lib.carbonpy_block_field_get_partition.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_CarbonBlockPartitionLayout),
        ]
        self.lib.carbonpy_block_field_get_partition.restype = ctypes.c_int

        self.lib.carbonpy_mgrid_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.carbonpy_mgrid_new.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_new_sparse.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.lib.carbonpy_mgrid_new_sparse.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_mgrid_delete.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_depth.argtypes = [self.handle_type]
        self.lib.carbonpy_mgrid_depth.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_field_new.argtypes = [
            ctypes.POINTER(self.handle_type),
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.carbonpy_mgrid_field_new.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_field_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.lib.carbonpy_mgrid_field_delete.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_get_span.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CarbonBlockSpan),
        ]
        self.lib.carbonpy_mgrid_get_span.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_field_get_partition.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_CarbonBlockPartitionLayout),
        ]
        self.lib.carbonpy_mgrid_field_get_partition.restype = ctypes.c_int
        self.lib.carbonpy_mgrid_field_get_cross_partition.argtypes = [
            self.handle_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_CarbonMgridPartitionLayout),
        ]
        self.lib.carbonpy_mgrid_field_get_cross_partition.restype = ctypes.c_int

        from .container import register_container_api

        register_container_api(self)

        from .skeleton import register_skeleton_api

        register_skeleton_api(self)
