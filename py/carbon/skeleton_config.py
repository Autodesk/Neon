"""Skeleton configuration mirroring Neon's ``SkeletonConfig`` (OCC overlap knob).

The integer values follow Neon's ``SkeletonConfig.OCC`` ordering and are mapped onto
``carbon::OverlapMode`` inside ``carbonpy_skeleton_compile``.
"""

from __future__ import annotations

from enum import Enum


class SkeletonConfig:
    """Container for skeleton configuration options."""

    class OCC:
        """Overlap-communication-computation strategy for a skeleton sequence."""

        class Values(Enum):
            standard = 0
            extended = 1
            twoWayExtended = 2
            none = 3

        def __init__(self, occ_config: "SkeletonConfig.OCC.Values") -> None:
            if not isinstance(occ_config, SkeletonConfig.OCC.Values):
                raise TypeError(f"expected OCC.Values, got {type(occ_config)!r}")
            self._value = occ_config

        @property
        def value(self) -> int:
            return int(self._value.value)

        def to_string(self) -> str:
            return self._value.name

        def __str__(self) -> str:
            return f"<OCC: {self._value.name}>"

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, SkeletonConfig.OCC):
                return NotImplemented
            return self._value == other._value

        def __hash__(self) -> int:
            return hash(self._value)

        @staticmethod
        def standard() -> "SkeletonConfig.OCC":
            return SkeletonConfig.OCC(SkeletonConfig.OCC.Values.standard)

        @staticmethod
        def extended() -> "SkeletonConfig.OCC":
            return SkeletonConfig.OCC(SkeletonConfig.OCC.Values.extended)

        @staticmethod
        def twoWayExtended() -> "SkeletonConfig.OCC":
            return SkeletonConfig.OCC(SkeletonConfig.OCC.Values.twoWayExtended)

        @staticmethod
        def none() -> "SkeletonConfig.OCC":
            return SkeletonConfig.OCC(SkeletonConfig.OCC.Values.none)

        @staticmethod
        def from_int(v: int) -> "SkeletonConfig.OCC":
            for member in SkeletonConfig.OCC.Values:
                if member.value == v:
                    return SkeletonConfig.OCC(member)
            raise ValueError(f"invalid OCC value: {v}")

        @staticmethod
        def from_string(s: str) -> "SkeletonConfig.OCC":
            try:
                return SkeletonConfig.OCC(SkeletonConfig.OCC.Values[s])
            except KeyError as exc:
                raise ValueError(
                    f"invalid OCC string: {s!r}; expected one of "
                    "standard, extended, twoWayExtended, none"
                ) from exc


OCC = SkeletonConfig.OCC
