"""A small 3D integer index, mirroring Neon's ``py_neon.Index_3d`` ergonomics.

Used at the Python level to describe grid extents (``Index_3d(nx, ny, nz)``); it is not
a device-blittable type (that is :class:`carbon.warp_types.CarbonIndex3`).
"""

from __future__ import annotations


class Index_3d:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: int, y: int, z: int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __eq__(self, other) -> bool:
        if isinstance(other, Index_3d):
            return (self.x, self.y, self.z) == (other.x, other.y, other.z)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Index_3d({self.x}, {self.y}, {self.z})"
