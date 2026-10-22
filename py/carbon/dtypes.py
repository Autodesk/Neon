"""Scalar dtype mapping shared by the object-oriented grid bindings.

Each Carbon field is specialized on a scalar type at creation. A Warp dtype
(``wp.float32`` / ``wp.float64`` / ``wp.int32`` / ``wp.int64``) maps to:

* a C-ABI code (``carbonpy::detail::CarbonDtype``) stored in the field box, and
* a Warp partition class name suffix (``f32`` / ``f64`` / ``i32`` / ``i64``) returned
  by ``field.get_partition()`` for ``wp.carbon.read`` / ``write`` overload resolution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DtypeInfo:
    code: int
    suffix: str


_STRING_ALIASES = {
    "f32": DtypeInfo(0, "f32"),
    "float32": DtypeInfo(0, "f32"),
    "f64": DtypeInfo(1, "f64"),
    "float64": DtypeInfo(1, "f64"),
    "i32": DtypeInfo(2, "i32"),
    "int32": DtypeInfo(2, "i32"),
    "i64": DtypeInfo(3, "i64"),
    "int64": DtypeInfo(3, "i64"),
    "u8": DtypeInfo(4, "u8"),
    "uint8": DtypeInfo(4, "u8"),
}


def _warp_aliases() -> dict:
    import warp as wp

    return {
        wp.float32: DtypeInfo(0, "f32"),
        wp.float64: DtypeInfo(1, "f64"),
        wp.int32: DtypeInfo(2, "i32"),
        wp.int64: DtypeInfo(3, "i64"),
        wp.uint8: DtypeInfo(4, "u8"),
    }


def resolve_dtype(dtype) -> DtypeInfo:
    """Map a Warp scalar type or string alias to its :class:`DtypeInfo`."""
    if isinstance(dtype, DtypeInfo):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _STRING_ALIASES:
            return _STRING_ALIASES[key]
        raise ValueError(f"unsupported carbon dtype string: {dtype!r}")
    try:
        aliases = _warp_aliases()
    except ImportError as exc:  # pragma: no cover - warp always present in use
        raise ValueError(
            f"cannot resolve dtype {dtype!r} without warp installed"
        ) from exc
    if dtype in aliases:
        return aliases[dtype]
    raise ValueError(
        f"unsupported carbon dtype: {dtype!r} "
        "(expected wp.float32/float64/int32/int64 or a string alias)"
    )
