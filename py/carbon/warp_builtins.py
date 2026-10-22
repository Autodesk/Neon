import os
import types

import warp as wp

from .warp_types import (
    BLOCK_PARTITION_BY_SUFFIX,
    BLOCK_PARTITION_TYPES,
    CarbonBlockIdx,
    CarbonBlockSpan,
    CarbonDenseIdx,
    CarbonDenseSpan,
    CarbonIndex3,
    CarbonProbeFatIdx,
    CarbonProbeValidatedIdx,
    DENSE_PARTITION_BY_SUFFIX,
    DENSE_PARTITION_TYPES,
    MGRID_PARTITION_BY_SUFFIX,
    MGRID_PARTITION_TYPES,
    NEIGHBOUR_VALUE_BY_SUFFIX,
    NEIGHBOUR_VALUE_TYPES,
    WARP_NEIGHBOUR_VALUE_NATIVE_NAMES,
    WARP_PARTITION_NATIVE_NAMES,
)

_CARBON_CPP_NAMESPACE = "wp::carbon::"

_SCALAR_DTYPES = (
    ("f32", wp.float32),
    ("f64", wp.float64),
    ("i32", wp.int32),
    ("i64", wp.int64),
)

# Dense grids additionally support uint8 (LBM masks: bc_mask / missing_mask / solid_mask).
# Block/Mgrid do not need it yet, so it is scoped to the dense-only registration paths.
_DENSE_DTYPES = _SCALAR_DTYPES + (("u8", wp.uint8),)

_CARBON_NAMESPACE_EXPORTS: dict[str, str] = {}


def _add_header(path: str) -> None:
    include_directive = f'#include "{path}"\n'
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_carbon_builtin(
    key: str,
    *,
    input_types: dict,
    value_type,
    native_func: str | None = None,
    export_name: str | None = None,
) -> None:
    wp.context.add_builtin(
        key,
        input_types=input_types,
        value_type=value_type,
        namespace=_CARBON_CPP_NAMESPACE,
        native_func=native_func if native_func is not None else key,
        export=False,
        missing_grad=True,
    )
    _CARBON_NAMESPACE_EXPORTS[export_name or key] = key


def _register_read_write_for_dtype(suffix: str, value_type) -> None:
    mgrid_part = MGRID_PARTITION_BY_SUFFIX[suffix]

    _register_carbon_builtin(
        "read_child",
        input_types={
            "part": mgrid_part,
            "idx": CarbonBlockIdx,
            "ox": int,
            "oy": int,
            "oz": int,
            "component_id": int,
        },
        value_type=value_type,
        native_func="read_child",
        export_name="read_child",
    )


def _register_mgrid_query(key: str, part_type: type, input_types: dict, value_type) -> None:
    inputs = {"part": part_type, **input_types}
    _register_carbon_builtin(
        key,
        input_types=inputs,
        value_type=value_type,
        export_name=key,
    )


def _register_idx_builtins() -> None:
    wp.types.add_type(CarbonIndex3, native_name="CarbonIndex3", has_binary_ctor=True)
    wp.types.add_type(CarbonDenseIdx, native_name="CarbonDenseIdx", has_binary_ctor=True)
    wp.types.add_type(CarbonBlockIdx, native_name="CarbonBlockIdx", has_binary_ctor=True)

    _register_carbon_builtin(
        "set",
        input_types={
            "span": CarbonDenseSpan,
            "ix": int,
            "iy": int,
            "iz": int,
        },
        value_type=CarbonDenseIdx,
    )
    _register_carbon_builtin(
        "set",
        input_types={
            "span": CarbonBlockSpan,
            "iter_idx": int,
            "x": int,
            "y": int,
            "z": int,
        },
        value_type=CarbonBlockIdx,
    )
    _register_carbon_builtin(
        "is_valid",
        input_types={"span": CarbonBlockSpan, "idx": CarbonBlockIdx},
        value_type=wp.bool,
    )
    _register_carbon_builtin(
        "get_x",
        input_types={"idx": CarbonIndex3},
        value_type=int,
    )
    _register_carbon_builtin(
        "get_y",
        input_types={"idx": CarbonIndex3},
        value_type=int,
    )
    _register_carbon_builtin(
        "get_z",
        input_types={"idx": CarbonIndex3},
        value_type=int,
    )

    # Dense read/write/global_idx cover the extra uint8 dtype (masks); Block/Mgrid stay on the
    # four numeric dtypes. global_idx(CarbonDenseSpan) is dtype-agnostic, so register it once.
    _register_carbon_builtin(
        "global_idx",
        input_types={"span": CarbonDenseSpan, "idx": CarbonDenseIdx},
        value_type=CarbonIndex3,
        export_name="global_idx",
    )
    for suffix, value_type in _DENSE_DTYPES:
        dense_part = DENSE_PARTITION_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "read",
            input_types={
                "part": dense_part,
                "span": CarbonDenseSpan,
                "idx": CarbonDenseIdx,
                "component_id": int,
            },
            value_type=value_type,
            native_func="read",
            export_name="read",
        )
        _register_carbon_builtin(
            "write",
            input_types={
                "part": dense_part,
                "span": CarbonDenseSpan,
                "idx": CarbonDenseIdx,
                "component_id": int,
                "value": value_type,
            },
            value_type=None,
            native_func="write",
            export_name="write",
        )

    _register_carbon_builtin(
        "global_idx",
        input_types={"span": CarbonBlockSpan, "idx": CarbonBlockIdx},
        value_type=CarbonIndex3,
        export_name="global_idx",
    )
    for suffix, value_type in _SCALAR_DTYPES:
        block_part = BLOCK_PARTITION_BY_SUFFIX[suffix]
        mgrid_part = MGRID_PARTITION_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "read",
            input_types={
                "part": mgrid_part,
                "own_span": CarbonBlockSpan,
                "idx": CarbonBlockIdx,
                "component_id": int,
            },
            value_type=value_type,
            native_func="read",
            export_name="read",
        )
        _register_carbon_builtin(
            "read",
            input_types={
                "part": block_part,
                "span": CarbonBlockSpan,
                "idx": CarbonBlockIdx,
                "component_id": int,
            },
            value_type=value_type,
            native_func="read",
            export_name="read",
        )
        _register_carbon_builtin(
            "write",
            input_types={
                "part": block_part,
                "span": CarbonBlockSpan,
                "idx": CarbonBlockIdx,
                "component_id": int,
                "value": value_type,
            },
            value_type=None,
            native_func="write",
            export_name="write",
        )


def _neighbour_offset_inputs(part_type: type, span_type: type, idx_type: type) -> dict:
    return {
        "part": part_type,
        "span": span_type,
        "idx": idx_type,
        "ox": int,
        "oy": int,
        "oz": int,
        "component_id": int,
    }


def _register_neighbour_builtins() -> None:
    # Dense neighbour ops (+ value_or/value/valid on the neighbour-value struct) cover the extra
    # uint8 dtype; Block/Mgrid neighbour ops stay on the four numeric dtypes.
    for suffix, value_type in _DENSE_DTYPES:
        nv_type = NEIGHBOUR_VALUE_BY_SUFFIX[suffix]
        dense_part = DENSE_PARTITION_BY_SUFFIX[suffix]
        dense_inputs = _neighbour_offset_inputs(
            dense_part, CarbonDenseSpan, CarbonDenseIdx
        )
        _register_carbon_builtin(
            "neighbour",
            input_types=dense_inputs,
            value_type=nv_type,
            native_func="neighbour",
            export_name="neighbour",
        )
        _register_carbon_builtin(
            "read_ngh",
            input_types={**dense_inputs, "alternative": value_type},
            value_type=value_type,
            native_func="read_ngh",
            export_name="read_ngh",
        )
        _register_carbon_builtin(
            "value_or",
            input_types={"nv": nv_type, "alternative": value_type},
            value_type=value_type,
            export_name="value_or",
        )
        _register_carbon_builtin(
            "value",
            input_types={"nv": nv_type},
            value_type=value_type,
            export_name="value",
        )
        _register_carbon_builtin(
            "valid",
            input_types={"nv": nv_type},
            value_type=wp.bool,
            export_name="valid",
        )

    for suffix, value_type in _SCALAR_DTYPES:
        nv_type = NEIGHBOUR_VALUE_BY_SUFFIX[suffix]
        block_part = BLOCK_PARTITION_BY_SUFFIX[suffix]
        mgrid_part = MGRID_PARTITION_BY_SUFFIX[suffix]
        block_inputs = _neighbour_offset_inputs(
            block_part, CarbonBlockSpan, CarbonBlockIdx
        )
        mgrid_inputs = _neighbour_offset_inputs(
            mgrid_part, CarbonBlockSpan, CarbonBlockIdx
        )
        for inputs in (block_inputs, mgrid_inputs):
            _register_carbon_builtin(
                "neighbour",
                input_types=inputs,
                value_type=nv_type,
                native_func="neighbour",
                export_name="neighbour",
            )
            _register_carbon_builtin(
                "read_ngh",
                input_types={**inputs, "alternative": value_type},
                value_type=value_type,
                native_func="read_ngh",
                export_name="read_ngh",
            )


def _register_dense_spanless_builtins() -> None:
    """Neon-parity dense ops that address memory from the span embedded in the partition."""
    for suffix, value_type in _DENSE_DTYPES:
        dense_part = DENSE_PARTITION_BY_SUFFIX[suffix]
        nv_type = NEIGHBOUR_VALUE_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "set",
            input_types={"part": dense_part, "ix": int, "iy": int, "iz": int},
            value_type=CarbonDenseIdx,
            export_name="set",
        )
        _register_carbon_builtin(
            "global_idx",
            input_types={"part": dense_part, "idx": CarbonDenseIdx},
            value_type=CarbonIndex3,
            export_name="global_idx",
        )
        _register_carbon_builtin(
            "read",
            input_types={"part": dense_part, "idx": CarbonDenseIdx, "component_id": int},
            value_type=value_type,
            native_func="read",
            export_name="read",
        )
        _register_carbon_builtin(
            "write",
            input_types={
                "part": dense_part,
                "idx": CarbonDenseIdx,
                "component_id": int,
                "value": value_type,
            },
            value_type=None,
            native_func="write",
            export_name="write",
        )
        _register_carbon_builtin(
            "neighbour",
            input_types={
                "part": dense_part,
                "idx": CarbonDenseIdx,
                "ox": int,
                "oy": int,
                "oz": int,
                "component_id": int,
            },
            value_type=nv_type,
            native_func="neighbour",
            export_name="neighbour",
        )
        _register_carbon_builtin(
            "read_ngh",
            input_types={
                "part": dense_part,
                "idx": CarbonDenseIdx,
                "ox": int,
                "oy": int,
                "oz": int,
                "component_id": int,
                "alternative": value_type,
            },
            value_type=value_type,
            native_func="read_ngh",
            export_name="read_ngh",
        )


def _register_mgrid_spanless_builtins() -> None:
    """Neon-parity own-level mgrid ops that address memory from the span embedded in the fat
    cross partition (``part.own_span``), so multires kernels call set/global_idx/read/write/
    neighbour/read_ngh with the partition alone. The 3-arg ``read(part, span, idx, c)`` overload
    stays the *parent* read; these spanless forms are the *own-level* ops."""
    for suffix, value_type in _SCALAR_DTYPES:
        mgrid_part = MGRID_PARTITION_BY_SUFFIX[suffix]
        nv_type = NEIGHBOUR_VALUE_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "set",
            input_types={"part": mgrid_part, "iter_idx": int, "x": int, "y": int, "z": int},
            value_type=CarbonBlockIdx,
            export_name="set",
        )
        _register_carbon_builtin(
            "global_idx",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx},
            value_type=CarbonIndex3,
            export_name="global_idx",
        )
        _register_carbon_builtin(
            "read",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx, "component_id": int},
            value_type=value_type,
            native_func="read",
            export_name="read",
        )
        _register_carbon_builtin(
            "write",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "component_id": int,
                "value": value_type,
            },
            value_type=None,
            native_func="write",
            export_name="write",
        )
        _register_carbon_builtin(
            "neighbour",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "ox": int,
                "oy": int,
                "oz": int,
                "component_id": int,
            },
            value_type=nv_type,
            native_func="neighbour",
            export_name="neighbour",
        )
        _register_carbon_builtin(
            "read_ngh",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "ox": int,
                "oy": int,
                "oz": int,
                "component_id": int,
                "alternative": value_type,
            },
            value_type=value_type,
            native_func="read_ngh",
            export_name="read_ngh",
        )


def _register_mgrid_lbm_builtins() -> None:
    """Multires LBM builtins on the fat cross partition (Neon mPartition parity).

    ``has_child`` / ``has_parent`` take a ``CarbonBlockIdx`` (own-level cell);
    ``has_finer_ngh`` / ``lbm_read_coarser_ngh`` / ``mres_lbm_store_op`` take a packed
    ``CarbonIndex3`` direction (from ``ngh_idx``). ``lbm_read_coarser_ngh`` writes its
    trailing ``bool`` (uncle validity) by reference, matching Neon's out-param shape."""
    for suffix, value_type in _SCALAR_DTYPES:
        mgrid_part = MGRID_PARTITION_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "has_child",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx},
            value_type=wp.bool,
            export_name="has_child",
        )
        _register_carbon_builtin(
            "has_parent",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx},
            value_type=wp.bool,
            export_name="has_parent",
        )
        _register_carbon_builtin(
            "has_finer_ngh",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx, "ngh": CarbonIndex3},
            value_type=wp.bool,
            export_name="has_finer_ngh",
        )
        _register_carbon_builtin(
            "lbm_read_coarser_ngh",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "ngh": CarbonIndex3,
                "card": int,
                "alternative": value_type,
                "valid": wp.bool,
            },
            value_type=value_type,
            export_name="lbm_read_coarser_ngh",
        )
        _register_carbon_builtin(
            "mres_lbm_store_op",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "card": int,
                "ngh": CarbonIndex3,
                "value": value_type,
            },
            value_type=None,
            export_name="mres_lbm_store_op",
        )


def register_neon_aliases() -> None:
    """Register ``wp.neon_*`` builtins as aliases of the spanless ``wp::carbon::`` dense ops.

    This lets XLB's NEON-path kernel bodies (which call ``wp.neon_read`` / ``wp.neon_write`` /
    ``wp.neon_read_ngh`` / ``wp.neon_ngh_idx`` / ``wp.neon_global_idx`` / ``wp.neon_get_{x,y,z}``)
    run unchanged against Carbon. Must be called after :func:`register_carbon_warp_types`.
    The multires builtins (``neon_has_child`` / ``neon_has_parent`` / ``neon_has_finer_ngh`` /
    ``neon_lbm_read_coarser_ngh`` / ``neon_mres_lbm_store_op``) are aliased onto the fat mgrid
    cross partition below. (``neon_level`` is referenced only in XLB comments/TODOs, not live
    kernel bodies, so it is deferred.)
    """
    for suffix, value_type in _DENSE_DTYPES:
        dense_part = DENSE_PARTITION_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "neon_read",
            input_types={"part": dense_part, "idx": CarbonDenseIdx, "component_id": int},
            value_type=value_type,
            native_func="read",
            export_name="neon_read",
        )
        _register_carbon_builtin(
            "neon_write",
            input_types={
                "part": dense_part,
                "idx": CarbonDenseIdx,
                "component_id": int,
                "value": value_type,
            },
            value_type=None,
            native_func="write",
            export_name="neon_write",
        )
        _register_carbon_builtin(
            "neon_global_idx",
            input_types={"part": dense_part, "idx": CarbonDenseIdx},
            value_type=CarbonIndex3,
            native_func="global_idx",
            export_name="neon_global_idx",
        )
        _register_carbon_builtin(
            "neon_read_ngh",
            input_types={
                "part": dense_part,
                "idx": CarbonDenseIdx,
                "ngh": CarbonIndex3,
                "component_id": int,
                "alternative": value_type,
                "is_valid": wp.bool,
            },
            value_type=value_type,
            native_func="read_ngh",
            export_name="neon_read_ngh",
        )

    _register_carbon_builtin(
        "neon_ngh_idx",
        input_types={"dx": wp.int8, "dy": wp.int8, "dz": wp.int8},
        value_type=CarbonIndex3,
        native_func="ngh_idx",
        export_name="neon_ngh_idx",
    )
    for key, native in (("neon_get_x", "get_x"), ("neon_get_y", "get_y"), ("neon_get_z", "get_z")):
        _register_carbon_builtin(
            key,
            input_types={"idx": CarbonIndex3},
            value_type=wp.int32,
            native_func=native,
            export_name=key,
        )

    # Multires aliases over the fat mgrid cross partition. neon_read / neon_write / neon_global_idx
    # / neon_read_ngh on an mgrid partition resolve to the *own-level* spanless ops (C.2); the
    # five multires-specific ones resolve to the C.3 primitives.
    for suffix, value_type in _SCALAR_DTYPES:
        mgrid_part = MGRID_PARTITION_BY_SUFFIX[suffix]
        _register_carbon_builtin(
            "neon_read",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx, "component_id": int},
            value_type=value_type,
            native_func="read",
            export_name="neon_read",
        )
        _register_carbon_builtin(
            "neon_write",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "component_id": int,
                "value": value_type,
            },
            value_type=None,
            native_func="write",
            export_name="neon_write",
        )
        _register_carbon_builtin(
            "neon_global_idx",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx},
            value_type=CarbonIndex3,
            native_func="global_idx",
            export_name="neon_global_idx",
        )
        _register_carbon_builtin(
            "neon_read_ngh",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "ngh": CarbonIndex3,
                "component_id": int,
                "alternative": value_type,
                "is_valid": wp.bool,
            },
            value_type=value_type,
            native_func="read_ngh",
            export_name="neon_read_ngh",
        )
        _register_carbon_builtin(
            "neon_has_child",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx},
            value_type=wp.bool,
            native_func="has_child",
            export_name="neon_has_child",
        )
        _register_carbon_builtin(
            "neon_has_parent",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx},
            value_type=wp.bool,
            native_func="has_parent",
            export_name="neon_has_parent",
        )
        _register_carbon_builtin(
            "neon_has_finer_ngh",
            input_types={"part": mgrid_part, "idx": CarbonBlockIdx, "ngh": CarbonIndex3},
            value_type=wp.bool,
            native_func="has_finer_ngh",
            export_name="neon_has_finer_ngh",
        )
        _register_carbon_builtin(
            "neon_lbm_read_coarser_ngh",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "ngh": CarbonIndex3,
                "card": int,
                "alternative": value_type,
                "valid": wp.bool,
            },
            value_type=value_type,
            native_func="lbm_read_coarser_ngh",
            export_name="neon_lbm_read_coarser_ngh",
        )
        _register_carbon_builtin(
            "neon_mres_lbm_store_op",
            input_types={
                "part": mgrid_part,
                "idx": CarbonBlockIdx,
                "card": int,
                "ngh": CarbonIndex3,
                "value": value_type,
            },
            value_type=None,
            native_func="mres_lbm_store_op",
            export_name="neon_mres_lbm_store_op",
        )

    for name in (
        "neon_read",
        "neon_write",
        "neon_global_idx",
        "neon_read_ngh",
        "neon_ngh_idx",
        "neon_get_x",
        "neon_get_y",
        "neon_get_z",
        "neon_has_child",
        "neon_has_parent",
        "neon_has_finer_ngh",
        "neon_lbm_read_coarser_ngh",
        "neon_mres_lbm_store_op",
    ):
        setattr(wp, name, wp.context.builtin_functions[name])


def _attach_carbon_namespace() -> None:
    carbon_ns = types.SimpleNamespace()
    for export_name, key in _CARBON_NAMESPACE_EXPORTS.items():
        setattr(carbon_ns, export_name, wp.context.builtin_functions[key])
    wp.carbon = carbon_ns


def _register_probe_builtins() -> None:
    """Spike scaffolding (phase 3): probe nested-registered-type returns. See warp_probe.h."""
    wp.types.add_type(CarbonProbeFatIdx, native_name="CarbonProbeFatIdx", has_binary_ctor=True)
    wp.types.add_type(
        CarbonProbeValidatedIdx,
        native_name="CarbonProbeValidatedIdx",
        has_binary_ctor=True,
    )

    _register_carbon_builtin(
        "probe_make_fat",
        input_types={"span": CarbonDenseSpan, "ix": int, "iy": int, "iz": int},
        value_type=CarbonProbeFatIdx,
        export_name="probe_make_fat",
    )
    _register_carbon_builtin(
        "probe_fat_read",
        input_types={
            "part": DENSE_PARTITION_BY_SUFFIX["f32"],
            "cell": CarbonProbeFatIdx,
            "component_id": int,
        },
        value_type=wp.float32,
        export_name="probe_fat_read",
    )
    _register_carbon_builtin(
        "probe_fat_write",
        input_types={
            "part": DENSE_PARTITION_BY_SUFFIX["f32"],
            "cell": CarbonProbeFatIdx,
            "component_id": int,
            "value": wp.float32,
        },
        value_type=None,
        export_name="probe_fat_write",
    )
    _register_carbon_builtin(
        "probe_fat_idx",
        input_types={"cell": CarbonProbeFatIdx},
        value_type=CarbonDenseIdx,
        export_name="probe_fat_idx",
    )
    _register_carbon_builtin(
        "probe_set_and_validate",
        input_types={"span": CarbonDenseSpan, "ix": int, "iy": int, "iz": int},
        value_type=CarbonProbeValidatedIdx,
        export_name="probe_set_and_validate",
    )
    _register_carbon_builtin(
        "probe_validated_valid",
        input_types={"v": CarbonProbeValidatedIdx},
        value_type=wp.bool,
        export_name="probe_validated_valid",
    )
    _register_carbon_builtin(
        "probe_validated_idx",
        input_types={"v": CarbonProbeValidatedIdx},
        value_type=CarbonDenseIdx,
        export_name="probe_validated_idx",
    )


def register_carbon_warp_types() -> None:
    include_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "libcarbonpy", "include")
    )
    _add_header(os.path.join(include_root, "carbonpy", "warp_partition_types.h"))
    _add_header(os.path.join(include_root, "carbonpy", "warp_dense.h"))
    _add_header(os.path.join(include_root, "carbonpy", "warp_block.h"))
    _add_header(os.path.join(include_root, "carbonpy", "warp_mgrid.h"))
    _add_header(os.path.join(include_root, "carbonpy", "warp_carbon.h"))
    _add_header(os.path.join(include_root, "carbonpy", "warp_probe.h"))

    wp.types.add_type(CarbonDenseSpan, native_name="CarbonDenseSpan", has_binary_ctor=True)
    wp.types.add_type(CarbonBlockSpan, native_name="CarbonBlockSpan", has_binary_ctor=True)

    for part_type in DENSE_PARTITION_TYPES + BLOCK_PARTITION_TYPES + MGRID_PARTITION_TYPES:
        wp.types.add_type(
            part_type,
            native_name=WARP_PARTITION_NATIVE_NAMES[part_type],
            has_binary_ctor=True,
        )

    for nv_type in NEIGHBOUR_VALUE_TYPES:
        wp.types.add_type(
            nv_type,
            native_name=WARP_NEIGHBOUR_VALUE_NATIVE_NAMES[nv_type],
            has_binary_ctor=True,
        )

    for suffix, value_type in _SCALAR_DTYPES:
        _register_read_write_for_dtype(suffix, value_type)

    _register_idx_builtins()
    _register_probe_builtins()
    _register_neighbour_builtins()
    _register_dense_spanless_builtins()
    _register_mgrid_spanless_builtins()
    _register_mgrid_lbm_builtins()

    _register_carbon_builtin(
        "view_block_idx",
        input_types={"span": CarbonBlockSpan, "iter_idx": int},
        value_type=int,
    )
    _register_carbon_builtin(
        "is_active",
        input_types={
            "span": CarbonBlockSpan,
            "block_idx": int,
            "x": int,
            "y": int,
            "z": int,
        },
        value_type=wp.bool,
    )
    _register_carbon_builtin(
        "global_x",
        input_types={
            "span": CarbonBlockSpan,
            "block_idx": int,
            "x": int,
            "y": int,
            "z": int,
        },
        value_type=int,
    )
    _register_carbon_builtin(
        "global_y",
        input_types={
            "span": CarbonBlockSpan,
            "block_idx": int,
            "x": int,
            "y": int,
            "z": int,
        },
        value_type=int,
    )
    _register_carbon_builtin(
        "global_z",
        input_types={
            "span": CarbonBlockSpan,
            "block_idx": int,
            "x": int,
            "y": int,
            "z": int,
        },
        value_type=int,
    )

    mgrid_query_specs = (
        (
            "has_parent",
            {"block_idx": int},
            wp.bool,
        ),
        (
            "has_children",
            {"block_idx": int, "x": int, "y": int, "z": int},
            wp.bool,
        ),
        (
            "parent_valid",
            {
                "own_span": CarbonBlockSpan,
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
            },
            wp.bool,
        ),
        (
            "child_valid",
            {
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
                "ox": int,
                "oy": int,
                "oz": int,
            },
            wp.bool,
        ),
        (
            "parent_global_x",
            {
                "own_span": CarbonBlockSpan,
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
            },
            int,
        ),
        (
            "parent_global_y",
            {
                "own_span": CarbonBlockSpan,
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
            },
            int,
        ),
        (
            "parent_global_z",
            {
                "own_span": CarbonBlockSpan,
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
            },
            int,
        ),
        (
            "child_global_x",
            {
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
                "ox": int,
                "oy": int,
                "oz": int,
            },
            int,
        ),
        (
            "child_global_y",
            {
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
                "ox": int,
                "oy": int,
                "oz": int,
            },
            int,
        ),
        (
            "child_global_z",
            {
                "block_idx": int,
                "x": int,
                "y": int,
                "z": int,
                "ox": int,
                "oy": int,
                "oz": int,
            },
            int,
        ),
    )
    for part_type in MGRID_PARTITION_TYPES:
        for key, inputs, value_type in mgrid_query_specs:
            _register_mgrid_query(key, part_type, inputs, value_type)

    _attach_carbon_namespace()
