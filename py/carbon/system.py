"""Object-oriented handle for a Carbon multi-GPU System."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gate import Gate


class System:
    """Owns a ``carbon::System`` handle; the root of the grid object graph."""

    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self.handle = gate.handle_type()
        if gate.lib.carbonpy_system_new(ctypes.byref(self.handle)) != 0:
            raise RuntimeError("carbonpy_system_new failed")

    @property
    def device_count(self) -> int:
        return self.gate.lib.carbonpy_device_count()

    def get_num_devices(self) -> int:
        """Neon-parity alias for the visible CUDA device count."""
        return self.gate.lib.carbonpy_device_count()

    def get_device_name(self, device: int) -> str:
        """Neon-parity device handle: the Warp device string for ``device``."""
        return f"cuda:{device}"

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_system_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "System":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()
