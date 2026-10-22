"""Loader token capture mirroring carbon::KernelLoader (Warp path)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, List, Optional


class AccessMode(IntEnum):
    Read = 0
    Write = 1
    ReadWrite = 2


class AccessPattern(IntEnum):
    Map = 0
    Stencil = 1


class Operation(Enum):
    map = 0
    stencil = 1
    stencil_up = 2
    stencil_down = 3


class Discretization(Enum):
    cartesian = 0
    lattice = 1


@dataclass
class Token:
    field: Any
    mode: AccessMode
    pattern: AccessPattern = AccessPattern.Map
    operation: Operation = Operation.map
    discretization: Discretization = Discretization.cartesian


def _pattern_for_operation(operation: Operation) -> AccessPattern:
    if operation == Operation.stencil:
        return AccessPattern.Stencil
    return AccessPattern.Map


@dataclass
class Loader:
    kernels: List[Callable[..., Any]] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    parsing: bool = False
    gpu_id: int = 0
    data_view: int = 0
    grid: Any = None
    mres_level: Optional[int] = None

    def set_grid(self, grid) -> None:
        self.grid = grid

    def set_mres_grid(self, grid, level: int) -> None:
        self.grid = grid
        self.mres_level = level

    def _retrieve_grid(self):
        return self.grid

    def get_mres_level(self) -> Optional[int]:
        return self.mres_level

    def get_device_id(self) -> int:
        """Neon-parity: index of the device this loading pass is capturing for."""
        return self.gpu_id

    def _record_token(
        self,
        field_obj,
        mode: AccessMode,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian,
    ) -> None:
        if self.parsing:
            self.tokens.append(
                Token(
                    field_obj,
                    mode,
                    _pattern_for_operation(operation),
                    operation,
                    discretization,
                )
            )

    def get_read_handle(
        self,
        field_obj,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian,
    ):
        self._record_token(field_obj, AccessMode.Read, operation, discretization)
        return self._partition_for_field(field_obj)

    def get_write_handle(
        self,
        field_obj,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian,
    ):
        self._record_token(field_obj, AccessMode.Write, operation, discretization)
        return self._partition_for_field(field_obj)

    def get_read_write_handle(
        self,
        field_obj,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian,
    ):
        self._record_token(field_obj, AccessMode.ReadWrite, operation, discretization)
        return self._partition_for_field(field_obj)

    def get_mres_read_handle(
        self,
        field_obj,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian,
    ):
        self._record_token(field_obj, AccessMode.Read, operation, discretization)
        return self._partition_for_field(field_obj)

    def get_mres_write_handle(
        self,
        field_obj,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian,
    ):
        self._record_token(field_obj, AccessMode.Write, operation, discretization)
        return self._partition_for_field(field_obj)

    def _partition_for_field(self, field_obj):
        if self.mres_level is not None:
            return field_obj.get_partition(self.mres_level, self.gpu_id, self.data_view)
        return field_obj.get_partition(self.gpu_id, self.data_view)

    def get_span(self):
        if self.grid is None:
            raise RuntimeError("loader.set_grid() must be called before get_span()")
        if self.mres_level is not None:
            return self.grid.get_span(self.mres_level, self.gpu_id, self.data_view)
        return self.grid.get_span(self.gpu_id, self.data_view)

    def read(self, field_obj, pattern: AccessPattern = AccessPattern.Map) -> None:
        if self.parsing:
            operation = Operation.stencil if pattern == AccessPattern.Stencil else Operation.map
            self._record_token(field_obj, AccessMode.Read, operation)

    def write(self, field_obj, pattern: AccessPattern = AccessPattern.Map) -> None:
        if self.parsing:
            operation = Operation.stencil if pattern == AccessPattern.Stencil else Operation.map
            self._record_token(field_obj, AccessMode.Write, operation)

    def read_write(self, field_obj, pattern: AccessPattern = AccessPattern.Map) -> None:
        if self.parsing:
            operation = Operation.stencil if pattern == AccessPattern.Stencil else Operation.map
            self._record_token(field_obj, AccessMode.ReadWrite, operation)

    def read_stencil(self, field_obj) -> None:
        self.read(field_obj, AccessPattern.Stencil)

    def declare_kernel(self, fn: Callable[..., Any]) -> None:
        self.kernels.append(fn)

    def _retrieve_compute_lambda(self) -> Callable[..., Any]:
        if not self.kernels:
            raise RuntimeError("loading_fn must declare at least one kernel")
        return self.kernels[0]


# Neon parity: expose the operation / discretization enums as attributes on the Loader class
# itself (Neon code writes ``neon.Loader.Operation.stencil`` / ``.Discretization.lattice``).
Loader.Operation = Operation
Loader.Discretization = Discretization
