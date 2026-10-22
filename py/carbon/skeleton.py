"""Ordered token-graph container sequences for the Warp Driver path.

Mirrors Neon's ``neon.skeleton.Skeleton`` API shape: ``Skeleton(system)`` then
``sequence(name, containers, occ=SkeletonConfig.OCC.none())`` then ``run()``.
Scheduling, auto-halo insertion, and OCC overlap are delegated to the C++
``carbon::Skeleton`` engine via foreign-launch Warp kernels.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Optional, Union

from .skeleton_config import SkeletonConfig

if TYPE_CHECKING:
    from .container import BlockContainer, DenseContainer, MgridContainer
    from .gate import Gate


class Skeleton:
    def __init__(self, system) -> None:
        self.system = system
        self.gate = system.gate
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_skeleton_new(
            ctypes.byref(self.handle), system.handle
        )
        if rc != 0:
            raise RuntimeError("carbonpy_skeleton_new failed")
        self.name = ""
        self.occ = SkeletonConfig.OCC.none()

    def add(self, container: Union["DenseContainer", "BlockContainer", "MgridContainer", "BoundKernel"]) -> None:
        from .container import BlockContainer, DenseContainer, MgridContainer
        from .kernel import BoundKernel

        if isinstance(container, BoundKernel):
            container = container.container

        if isinstance(container, DenseContainer):
            rc = self.gate.lib.carbonpy_skeleton_add_dense_container(
                self.handle, container.handle
            )
        elif isinstance(container, BlockContainer):
            rc = self.gate.lib.carbonpy_skeleton_add_block_container(
                self.handle, container.handle
            )
        elif isinstance(container, MgridContainer):
            rc = self.gate.lib.carbonpy_skeleton_add_mgrid_container(
                self.handle, container.handle
            )
        else:
            raise TypeError(f"unsupported container type: {type(container)!r}")
        if rc != 0:
            raise RuntimeError("carbonpy_skeleton_add_* failed")

    def compile(self) -> None:
        rc = self.gate.lib.carbonpy_skeleton_compile(self.handle, self.occ.value)
        if rc != 0:
            raise RuntimeError("carbonpy_skeleton_compile failed")

    def sequence(
        self,
        name: str,
        containers,
        occ: Optional[SkeletonConfig.OCC] = None,
    ) -> "Skeleton":
        self.clear()
        self.name = name
        self.occ = occ if occ is not None else SkeletonConfig.OCC.none()
        for container in containers:
            self.add(container)
        self.compile()
        return self

    def inserted_halo_count(self) -> int:
        count = self.gate.lib.carbonpy_skeleton_inserted_halo_count(self.handle)
        if count < 0:
            raise RuntimeError("carbonpy_skeleton_inserted_halo_count failed")
        return count

    def ioToDot(self, filename: str, graph_name: str = "skeleton") -> None:
        rc = self.gate.lib.carbonpy_skeleton_export_dot(
            self.handle,
            filename.encode("utf-8"),
            graph_name.encode("utf-8"),
        )
        if rc != 0:
            raise RuntimeError("carbonpy_skeleton_export_dot failed")

    def kernel_streams(self) -> list[int]:
        count = self.gate.lib.carbonpy_skeleton_kernel_stream_count(self.handle)
        if count < 0:
            raise RuntimeError("carbonpy_skeleton_kernel_stream_count failed")
        streams: list[int] = []
        for i in range(count):
            stream = self.gate.lib.carbonpy_skeleton_kernel_stream_at(self.handle, i)
            if stream < 0:
                raise RuntimeError("carbonpy_skeleton_kernel_stream_at failed")
            streams.append(stream)
        return streams

    def clear(self) -> None:
        self.name = ""
        self.occ = SkeletonConfig.OCC.none()
        rc = self.gate.lib.carbonpy_skeleton_clear(self.handle)
        if rc != 0:
            raise RuntimeError("carbonpy_skeleton_clear failed")

    def run(self, stream_idx: int = 0, data_view: int = 0) -> None:
        rc = self.gate.lib.carbonpy_skeleton_run(self.handle, stream_idx, data_view)
        if rc != 0:
            raise RuntimeError(f"carbonpy_skeleton_run failed: {rc}")

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_skeleton_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "Skeleton":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


def register_skeleton_api(gate: Gate) -> None:
    gate.lib.carbonpy_skeleton_new.argtypes = [
        ctypes.POINTER(gate.handle_type),
        gate.handle_type,
    ]
    gate.lib.carbonpy_skeleton_new.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_delete.argtypes = [ctypes.POINTER(gate.handle_type)]
    gate.lib.carbonpy_skeleton_delete.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_clear.argtypes = [gate.handle_type]
    gate.lib.carbonpy_skeleton_clear.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_compile.argtypes = [
        gate.handle_type,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_skeleton_compile.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_inserted_halo_count.argtypes = [gate.handle_type]
    gate.lib.carbonpy_skeleton_inserted_halo_count.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_export_dot.argtypes = [
        gate.handle_type,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    gate.lib.carbonpy_skeleton_export_dot.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_add_dense_container.argtypes = [
        gate.handle_type,
        gate.handle_type,
    ]
    gate.lib.carbonpy_skeleton_add_dense_container.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_add_block_container.argtypes = [
        gate.handle_type,
        gate.handle_type,
    ]
    gate.lib.carbonpy_skeleton_add_block_container.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_add_mgrid_container.argtypes = [
        gate.handle_type,
        gate.handle_type,
    ]
    gate.lib.carbonpy_skeleton_add_mgrid_container.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_run.argtypes = [
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_skeleton_run.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_kernel_stream_count.argtypes = [gate.handle_type]
    gate.lib.carbonpy_skeleton_kernel_stream_count.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_kernel_stream_at.argtypes = [
        gate.handle_type,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_skeleton_kernel_stream_at.restype = ctypes.c_int
    gate.lib.carbonpy_skeleton_num_streams.argtypes = [gate.handle_type]
    gate.lib.carbonpy_skeleton_num_streams.restype = ctypes.c_int

    gate.lib.carbonpy_dense_container_add_token.argtypes = [
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_dense_container_add_token.restype = ctypes.c_int
    gate.lib.carbonpy_block_container_add_token.argtypes = [
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_block_container_add_token.restype = ctypes.c_int
    gate.lib.carbonpy_mgrid_container_add_token.argtypes = [
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_mgrid_container_add_token.restype = ctypes.c_int
