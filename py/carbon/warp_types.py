import ctypes


class CarbonDenseSpan(ctypes.Structure):
    _fields_ = [
        ("iter_x", ctypes.c_int64),
        ("iter_y", ctypes.c_int64),
        ("iter_z", ctypes.c_int64),
        ("logic_x", ctypes.c_int64),
        ("logic_y", ctypes.c_int64),
        ("logic_z", ctypes.c_int64),
        ("pitch_xy", ctypes.c_int64),
        ("z_offset", ctypes.c_int64),
        ("halo_x", ctypes.c_int32),
        ("halo_y", ctypes.c_int32),
        ("halo_z", ctypes.c_int32),
        ("global_x", ctypes.c_int64),
        ("global_y", ctypes.c_int64),
        ("global_z", ctypes.c_int64),
        ("boundary_mode", ctypes.c_int32),
        ("data_view", ctypes.c_int32),
    ]


class _CarbonDensePartitionLayout(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_uint64),
        ("span", CarbonDenseSpan),
    ]


class CarbonDensePartitionF32(_CarbonDensePartitionLayout):
    pass


class CarbonDensePartitionF64(_CarbonDensePartitionLayout):
    pass


class CarbonDensePartitionI32(_CarbonDensePartitionLayout):
    pass


class CarbonDensePartitionI64(_CarbonDensePartitionLayout):
    pass


class CarbonDensePartitionU8(_CarbonDensePartitionLayout):
    pass


CarbonDensePartition = CarbonDensePartitionF32


class CarbonBlockSpan(ctypes.Structure):
    _fields_ = [
        ("bitmask_ptr", ctypes.c_uint64),
        ("connectivity_ptr", ctypes.c_uint64),
        ("origins_ptr", ctypes.c_uint64),
        ("block_count", ctypes.c_uint32),
        ("n_internal", ctypes.c_uint32),
        ("iter_block_count", ctypes.c_uint32),
        ("mem_x", ctypes.c_int32),
        ("mem_y", ctypes.c_int32),
        ("mem_z", ctypes.c_int32),
    ]


class _CarbonBlockPartitionLayout(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_uint64),
        ("components", ctypes.c_int32),
    ]


class CarbonBlockPartitionF32(_CarbonBlockPartitionLayout):
    pass


class CarbonBlockPartitionF64(_CarbonBlockPartitionLayout):
    pass


class CarbonBlockPartitionI32(_CarbonBlockPartitionLayout):
    pass


class CarbonBlockPartitionI64(_CarbonBlockPartitionLayout):
    pass


CarbonBlockPartition = CarbonBlockPartitionF32


class _CarbonMgridPartitionLayout(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_uint64),
        ("ref_factor", ctypes.c_int32),
        ("components", ctypes.c_int32),
        ("own_span", CarbonBlockSpan),
        ("parent_data_ptr", ctypes.c_uint64),
        ("parent_block_id_ptr", ctypes.c_uint64),
        ("parent_span", CarbonBlockSpan),
        ("child_data_ptr", ctypes.c_uint64),
        ("child_block_id_ptr", ctypes.c_uint64),
        ("child_span", CarbonBlockSpan),
    ]


class CarbonMgridPartitionF32(_CarbonMgridPartitionLayout):
    pass


class CarbonMgridPartitionF64(_CarbonMgridPartitionLayout):
    pass


class CarbonMgridPartitionI32(_CarbonMgridPartitionLayout):
    pass


class CarbonMgridPartitionI64(_CarbonMgridPartitionLayout):
    pass


CarbonMgridPartition = CarbonMgridPartitionF32


class CarbonIndex3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int64),
        ("y", ctypes.c_int64),
        ("z", ctypes.c_int64),
    ]


class CarbonDenseIdx(ctypes.Structure):
    _fields_ = [
        ("ix", ctypes.c_int64),
        ("iy", ctypes.c_int64),
        ("iz", ctypes.c_int64),
    ]


class CarbonBlockIdx(ctypes.Structure):
    _fields_ = [
        ("block_idx", ctypes.c_uint32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("z", ctypes.c_int32),
    ]


class CarbonNeighbourValueF32(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_float),
        ("valid", ctypes.c_int32),
    ]


class CarbonNeighbourValueF64(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_double),
        ("valid", ctypes.c_int32),
    ]


class CarbonNeighbourValueI32(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_int32),
        ("valid", ctypes.c_int32),
    ]


class CarbonNeighbourValueI64(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_int64),
        ("valid", ctypes.c_int32),
    ]


class CarbonNeighbourValueU8(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint8),
        ("valid", ctypes.c_int32),
    ]


CarbonNeighbourValue = CarbonNeighbourValueF32


# Spike scaffolding for the two-layer binding plan (phase 3): probes whether a builtin can
# return a registered type that nests another registered type. Mirrors warp_probe.h. Delete
# together with that header once the question is settled and the real types land.
class CarbonProbeFatIdx(ctypes.Structure):
    _fields_ = [
        ("idx", CarbonDenseIdx),
        ("span", CarbonDenseSpan),
    ]


class CarbonProbeValidatedIdx(ctypes.Structure):
    _fields_ = [
        ("idx", CarbonDenseIdx),
        ("valid", ctypes.c_int32),
    ]


DENSE_PARTITION_BY_SUFFIX = {
    "f32": CarbonDensePartitionF32,
    "f64": CarbonDensePartitionF64,
    "i32": CarbonDensePartitionI32,
    "i64": CarbonDensePartitionI64,
    "u8": CarbonDensePartitionU8,
}

BLOCK_PARTITION_BY_SUFFIX = {
    "f32": CarbonBlockPartitionF32,
    "f64": CarbonBlockPartitionF64,
    "i32": CarbonBlockPartitionI32,
    "i64": CarbonBlockPartitionI64,
}

MGRID_PARTITION_BY_SUFFIX = {
    "f32": CarbonMgridPartitionF32,
    "f64": CarbonMgridPartitionF64,
    "i32": CarbonMgridPartitionI32,
    "i64": CarbonMgridPartitionI64,
}

DENSE_PARTITION_TYPES = tuple(DENSE_PARTITION_BY_SUFFIX.values())
BLOCK_PARTITION_TYPES = tuple(BLOCK_PARTITION_BY_SUFFIX.values())
MGRID_PARTITION_TYPES = tuple(MGRID_PARTITION_BY_SUFFIX.values())

NEIGHBOUR_VALUE_BY_SUFFIX = {
    "f32": CarbonNeighbourValueF32,
    "f64": CarbonNeighbourValueF64,
    "i32": CarbonNeighbourValueI32,
    "i64": CarbonNeighbourValueI64,
    "u8": CarbonNeighbourValueU8,
}

NEIGHBOUR_VALUE_TYPES = tuple(NEIGHBOUR_VALUE_BY_SUFFIX.values())

WARP_NEIGHBOUR_VALUE_NATIVE_NAMES = {
    CarbonNeighbourValueF32: "CarbonNeighbourValueF32",
    CarbonNeighbourValueF64: "CarbonNeighbourValueF64",
    CarbonNeighbourValueI32: "CarbonNeighbourValueI32",
    CarbonNeighbourValueI64: "CarbonNeighbourValueI64",
    CarbonNeighbourValueU8: "CarbonNeighbourValueU8",
}

WARP_PARTITION_NATIVE_NAMES = {
    CarbonDensePartitionF32: "CarbonDensePartitionF32",
    CarbonDensePartitionF64: "CarbonDensePartitionF64",
    CarbonDensePartitionI32: "CarbonDensePartitionI32",
    CarbonDensePartitionI64: "CarbonDensePartitionI64",
    CarbonDensePartitionU8: "CarbonDensePartitionU8",
    CarbonBlockPartitionF32: "CarbonBlockPartitionF32",
    CarbonBlockPartitionF64: "CarbonBlockPartitionF64",
    CarbonBlockPartitionI32: "CarbonBlockPartitionI32",
    CarbonBlockPartitionI64: "CarbonBlockPartitionI64",
    CarbonMgridPartitionF32: "CarbonMgridPartitionF32",
    CarbonMgridPartitionF64: "CarbonMgridPartitionF64",
    CarbonMgridPartitionI32: "CarbonMgridPartitionI32",
    CarbonMgridPartitionI64: "CarbonMgridPartitionI64",
}
