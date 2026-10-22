"""Object-oriented Block (block-sparse) grid bindings for Carbon + NVIDIA Warp."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from ..dtypes import resolve_dtype

if TYPE_CHECKING:
    from ..system import System


class BlockField:
    """A Block field specialized on a scalar dtype and component count."""

    def __init__(self, block: "Block", components: int, dtype) -> None:
        self.block = block
        self.gate = block.gate
        self.components = components
        info = resolve_dtype(dtype)
        self.dtype = dtype
        self.dtype_code = info.code
        self.builtin_suffix = info.suffix
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_block_field_new(
            ctypes.byref(self.handle), block.handle, components, info.code
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_field_new failed")

    def get_grid(self) -> "Block":
        return self.block

    def get_partition(self, device: int = 0, data_view: int = 0):
        from ..warp_types import BLOCK_PARTITION_BY_SUFFIX

        part = BLOCK_PARTITION_BY_SUFFIX[self.builtin_suffix]()
        rc = self.gate.lib.carbonpy_block_field_get_partition(
            self.handle, device, data_view, ctypes.byref(part)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_field_get_partition failed")
        return part

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_block_field_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "BlockField":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


class Block:
    """A block-sparse grid (port of Neon's bGrid) over a :class:`System`."""

    name = "Block"

    def __init__(
        self, system: "System", extent_x: int, extent_y: int, extent_z: int
    ) -> None:
        self.system = system
        self.gate = system.gate
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_block_new(
            ctypes.byref(self.handle),
            system.handle,
            ctypes.c_int64(extent_x),
            ctypes.c_int64(extent_y),
            ctypes.c_int64(extent_z),
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_new failed")

    def new_field(self, components: int = 1, dtype=None) -> BlockField:
        if dtype is None:
            import warp as wp

            dtype = wp.float32
        return BlockField(self, components, dtype)

    def get_span(self, device: int = 0, data_view: int = 0):
        span = self.gate.CarbonBlockSpan()
        rc = self.gate.lib.carbonpy_block_get_span(
            self.handle, device, data_view, ctypes.byref(span)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_get_span failed")
        return span

    def kernel(
        self,
        driver,
        field,
        *,
        name: str = "carbon_kernel",
        access=None,
    ):
        from ..kernel import block_kernel

        return block_kernel(driver, self, field, name=name, access=access)

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_block_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "Block":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()
