"""CUDA Driver + Dense container launch via libcarbonpy."""

from __future__ import annotations

import ctypes
from typing import Callable, TYPE_CHECKING

import warp as wp

from .warp_types import (
    CarbonBlockPartition,
    CarbonBlockSpan,
    CarbonDensePartition,
    CarbonDenseSpan,
)

# Carbon kernels are forward-only (Carbon has no autodiff), so we disable Warp's
# backward-pass codegen for this module. Beyond faster compiles, this is required
# for correctness: index handles such as CarbonIndex3 are returned by neon_* funcs
# (e.g. the boundary-masker's get_pull_index), and Warp's adjoint pass would emit a
# `+= ` on CarbonIndex3 — a type with no wp::add — which fails to compile.
wp.set_module_options({"enable_backward": False})

if TYPE_CHECKING:
    from .gate import Gate


class LaunchBounds(ctypes.Structure):
    _fields_ = [
        ("shape", ctypes.c_int32 * 4),
        ("ndim", ctypes.c_int32),
        ("size", ctypes.c_size_t),
    ]


def _handle_of(obj):
    """Accept an OO grid/field object or a raw ctypes handle interchangeably."""
    return getattr(obj, "handle", obj)


class CudaDriver:
    def __init__(self, system) -> None:
        self.system = system
        self.gate = system.gate
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_cuda_driver_new(
            ctypes.byref(self.handle), system.handle
        )
        if rc != 0:
            raise RuntimeError("carbonpy_cuda_driver_new failed")

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_cuda_driver_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "CudaDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


class DenseContainer:
    DATA_VIEW_COUNT = 3

    def __init__(
        self,
        driver: CudaDriver,
        dense,
        loading_fn: Callable,
        name: str = "carbon_container",
        field_count: int = 1,
        *,
        capture: bool = False,
    ) -> None:
        gate = driver.gate
        self.gate = gate
        self.driver = driver
        self.dense = dense
        self.dense_handle = dense.handle
        dense_handle = dense.handle
        self.name = name
        self.field_count = field_count
        self.capture = capture

        n_devices = gate.lib.carbonpy_device_count()
        self._modules: list[set] = [set() for _ in range(n_devices)]
        hooks = (ctypes.c_void_p * (n_devices * self.DATA_VIEW_COUNT))()

        for gpu in range(n_devices):
            for view in range(self.DATA_VIEW_COUNT):
                kernel = self._build_kernel(
                    gate, loading_fn, gpu, view, field_count, self.capture
                )
                hook = self._kernel_hook(kernel, gpu)
                hooks[gpu * self.DATA_VIEW_COUNT + view] = hook

        self.handle = gate.handle_type()
        rc = gate.lib.carbonpy_dense_container_new(
            ctypes.byref(self.handle),
            driver.handle,
            dense_handle,
            hooks,
            len(hooks),
            field_count,
            1 if self.capture else 0,
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_container_new failed")

        self._register_loader_tokens(loading_fn, self._add_token)

    def _add_token(self, field, mode: int, pattern: int) -> None:
        rc = self.gate.lib.carbonpy_dense_container_add_token(
            self.handle, _handle_of(field), mode, pattern
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_container_add_token failed")

    def _register_loader_tokens(self, loading_fn: Callable, add_token_fn) -> None:
        from .loader import Loader

        loader = Loader(parsing=True)

        def bind_field(field, field_slot: int = 0):
            if field is None:
                raise RuntimeError("container loading_fn must bind a field")

        loading_fn(loader, bind_field)
        for token in loader.tokens:
            add_token_fn(token.field, int(token.mode), int(token.pattern))

    def bind_field(self, field, device: int = 0, field_slot: int = 0) -> None:
        rc = self.gate.lib.carbonpy_dense_container_bind_field(
            self.handle, _handle_of(field), device, field_slot
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_container_bind_field failed")

    def run(self, stream_idx: int = 0, data_view: int = 0) -> None:
        rc = self.gate.lib.carbonpy_dense_container_run(
            self.handle, stream_idx, data_view
        )
        if rc != 0:
            raise RuntimeError(f"carbonpy_dense_container_run failed: {rc}")

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_dense_container_delete(ctypes.byref(self.handle))

    def _build_kernel(
        self,
        gate: Gate,
        loading_fn: Callable,
        gpu_id: int,
        data_view: int,
        field_count: int,
        capture: bool = False,
    ):
        from .loader import Loader

        loader = Loader(gpu_id=gpu_id, data_view=data_view)

        def bind_field(field_handle, field_slot: int = 0):
            if field_handle is None:
                raise RuntimeError("container loading_fn must bind a field")

        loading_fn(loader, bind_field)

        if not loader.kernels:
            raise RuntimeError("container loading_fn must declare at least one kernel")

        compute_fn = loader.kernels[0]

        if capture:
            span = loader.get_span()

            @wp.kernel
            def kernel():
                ix, iy, iz = wp.tid()
                idx = wp.carbon.set(span, ix, iy, iz)
                compute_fn(idx)

            return kernel

        if field_count == 1:

            @wp.kernel
            def kernel(part: CarbonDensePartition, span: CarbonDenseSpan):
                ix, iy, iz = wp.tid()
                compute_fn(part, span, ix, iy, iz)

        elif field_count == 2:

            @wp.kernel
            def kernel(
                part0: CarbonDensePartition,
                span0: CarbonDenseSpan,
                part1: CarbonDensePartition,
                span1: CarbonDenseSpan,
            ):
                ix, iy, iz = wp.tid()
                compute_fn(part0, span0, part1, span1, ix, iy, iz)

        else:
            raise RuntimeError(f"unsupported dense container field_count={field_count}")

        return kernel

    def _kernel_hook(self, kernel, gpu_id: int):
        device = wp.get_device(f"cuda:{gpu_id}")
        module_exec = kernel.module.load(device)
        if module_exec is None:
            raise RuntimeError(f"failed to load Warp module on cuda:{gpu_id}")
        self._modules[gpu_id].add(module_exec)
        return module_exec.get_kernel_hooks(kernel).forward


class BlockContainer:
    DATA_VIEW_COUNT = 3

    def __init__(
        self,
        driver: CudaDriver,
        block,
        loading_fn: Callable,
        name: str = "carbon_block_container",
        *,
        capture: bool = False,
    ) -> None:
        gate = driver.gate
        self.gate = gate
        self.driver = driver
        self.block = block
        self.block_handle = block.handle
        block_handle = block.handle
        self.name = name
        self.capture = capture

        n_devices = gate.lib.carbonpy_device_count()
        self._modules: list[set] = [set() for _ in range(n_devices)]
        hooks = (ctypes.c_void_p * (n_devices * self.DATA_VIEW_COUNT))()

        for gpu in range(n_devices):
            for view in range(self.DATA_VIEW_COUNT):
                kernel = self._build_kernel(loading_fn, gpu, view, self.capture)
                hook = self._kernel_hook(kernel, gpu)
                hooks[gpu * self.DATA_VIEW_COUNT + view] = hook

        self.handle = gate.handle_type()
        rc = gate.lib.carbonpy_block_container_new(
            ctypes.byref(self.handle),
            driver.handle,
            block_handle,
            hooks,
            len(hooks),
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_container_new failed")

        self._register_loader_tokens(loading_fn, self._add_token)

    def _add_token(self, field, mode: int, pattern: int) -> None:
        rc = self.gate.lib.carbonpy_block_container_add_token(
            self.handle, _handle_of(field), mode, pattern
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_container_add_token failed")

    def _register_loader_tokens(self, loading_fn: Callable, add_token_fn) -> None:
        from .loader import Loader

        loader = Loader(parsing=True)

        def bind_field(field):
            if field is None:
                raise RuntimeError("container loading_fn must bind a field")

        loading_fn(loader, bind_field)
        for token in loader.tokens:
            add_token_fn(token.field, int(token.mode), int(token.pattern))

    def bind_field(self, field, device: int = 0) -> None:
        rc = self.gate.lib.carbonpy_block_container_bind_field(
            self.handle, _handle_of(field), device
        )
        if rc != 0:
            raise RuntimeError("carbonpy_block_container_bind_field failed")

    def run(self, stream_idx: int = 0, data_view: int = 0) -> None:
        rc = self.gate.lib.carbonpy_block_container_run(
            self.handle, stream_idx, data_view
        )
        if rc != 0:
            raise RuntimeError(f"carbonpy_block_container_run failed: {rc}")

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_block_container_delete(ctypes.byref(self.handle))

    def _build_kernel(
        self, loading_fn: Callable, gpu_id: int, data_view: int, capture: bool = False
    ):
        from .loader import Loader

        loader = Loader(gpu_id=gpu_id, data_view=data_view)

        def bind_field(field_handle):
            if field_handle is None:
                raise RuntimeError("container loading_fn must bind a field")

        loading_fn(loader, bind_field)

        if not loader.kernels:
            raise RuntimeError("container loading_fn must declare at least one kernel")

        compute_fn = loader.kernels[0]

        if capture:
            span = loader.get_span()

            @wp.kernel
            def kernel():
                iter_idx, x, y, z = wp.tid()
                idx = wp.carbon.set(span, iter_idx, x, y, z)
                if wp.carbon.is_valid(span, idx):
                    compute_fn(idx)

            return kernel

        @wp.kernel
        def kernel(part: CarbonBlockPartition, span: CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            compute_fn(part, span, iter_idx, x, y, z)

        return kernel

    def _kernel_hook(self, kernel, gpu_id: int):
        device = wp.get_device(f"cuda:{gpu_id}")
        module_exec = kernel.module.load(device)
        if module_exec is None:
            raise RuntimeError(f"failed to load Warp module on cuda:{gpu_id}")
        self._modules[gpu_id].add(module_exec)
        return module_exec.get_kernel_hooks(kernel).forward


class MgridContainer:
    DATA_VIEW_COUNT = 3

    def __init__(
        self,
        driver: CudaDriver,
        mgrid,
        level: int,
        loading_fn: Callable,
        name: str = "carbon_mgrid_container",
        *,
        capture: bool = False,
    ) -> None:
        gate = driver.gate
        self.gate = gate
        self.driver = driver
        self.mgrid = mgrid
        self.mgrid_handle = mgrid.handle
        mgrid_handle = mgrid.handle
        self.level = level
        self.name = name
        self.capture = capture

        n_devices = gate.lib.carbonpy_device_count()
        self._modules: list[set] = [set() for _ in range(n_devices)]
        hooks = (ctypes.c_void_p * (n_devices * self.DATA_VIEW_COUNT))()

        for gpu in range(n_devices):
            for view in range(self.DATA_VIEW_COUNT):
                kernel = self._build_kernel(loading_fn, gpu, view, self.capture)
                hook = self._kernel_hook(kernel, gpu)
                hooks[gpu * self.DATA_VIEW_COUNT + view] = hook

        self.handle = gate.handle_type()
        rc = gate.lib.carbonpy_mgrid_container_new(
            ctypes.byref(self.handle),
            driver.handle,
            mgrid_handle,
            level,
            hooks,
            len(hooks),
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_container_new failed")

        self._register_loader_tokens(loading_fn, self._add_token)

    def _add_token(self, field, mode: int, pattern: int) -> None:
        rc = self.gate.lib.carbonpy_mgrid_container_add_token(
            self.handle, _handle_of(field), mode, pattern
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_container_add_token failed")

    def _register_loader_tokens(self, loading_fn: Callable, add_token_fn) -> None:
        from .loader import Loader

        loader = Loader(parsing=True)

        def bind_field(field):
            if field is None:
                raise RuntimeError("container loading_fn must bind a field")

        loading_fn(loader, bind_field)
        for token in loader.tokens:
            add_token_fn(token.field, int(token.mode), int(token.pattern))

    def bind_field(self, field, device: int = 0) -> None:
        rc = self.gate.lib.carbonpy_mgrid_container_bind_field(
            self.handle, _handle_of(field), device
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_container_bind_field failed")

    def run(self, stream_idx: int = 0, data_view: int = 0) -> None:
        rc = self.gate.lib.carbonpy_mgrid_container_run(
            self.handle, stream_idx, data_view
        )
        if rc != 0:
            raise RuntimeError(f"carbonpy_mgrid_container_run failed: {rc}")

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_mgrid_container_delete(ctypes.byref(self.handle))

    def _build_kernel(
        self, loading_fn: Callable, gpu_id: int, data_view: int, capture: bool = False
    ):
        from .loader import Loader

        loader = Loader(gpu_id=gpu_id, data_view=data_view)

        def bind_field(field_handle):
            if field_handle is None:
                raise RuntimeError("container loading_fn must bind a field")

        loading_fn(loader, bind_field)

        if not loader.kernels:
            raise RuntimeError("container loading_fn must declare at least one kernel")

        compute_fn = loader.kernels[0]

        if capture:
            span = loader.get_span()

            @wp.kernel
            def kernel():
                iter_idx, x, y, z = wp.tid()
                idx = wp.carbon.set(span, iter_idx, x, y, z)
                if wp.carbon.is_valid(span, idx):
                    compute_fn(idx)

            return kernel

        @wp.kernel
        def kernel(part: CarbonBlockPartition, span: CarbonBlockSpan):
            iter_idx, x, y, z = wp.tid()
            compute_fn(part, span, iter_idx, x, y, z)

        return kernel

    def _kernel_hook(self, kernel, gpu_id: int):
        device = wp.get_device(f"cuda:{gpu_id}")
        module_exec = kernel.module.load(device)
        if module_exec is None:
            raise RuntimeError(f"failed to load Warp module on cuda:{gpu_id}")
        self._modules[gpu_id].add(module_exec)
        return module_exec.get_kernel_hooks(kernel).forward


def register_container_api(gate: Gate) -> None:
    gate.LaunchBounds = LaunchBounds
    gate.lib.carbonpy_cuda_driver_new.argtypes = [
        ctypes.POINTER(gate.handle_type),
        gate.handle_type,
    ]
    gate.lib.carbonpy_cuda_driver_new.restype = ctypes.c_int
    gate.lib.carbonpy_cuda_driver_delete.argtypes = [ctypes.POINTER(gate.handle_type)]
    gate.lib.carbonpy_cuda_driver_delete.restype = ctypes.c_int

    gate.lib.carbonpy_dense_container_new.argtypes = [
        ctypes.POINTER(gate.handle_type),
        gate.handle_type,
        gate.handle_type,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_dense_container_new.restype = ctypes.c_int
    gate.lib.carbonpy_dense_container_delete.argtypes = [ctypes.POINTER(gate.handle_type)]
    gate.lib.carbonpy_dense_container_delete.restype = ctypes.c_int
    gate.lib.carbonpy_dense_container_bind_field.argtypes = [
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_dense_container_bind_field.restype = ctypes.c_int
    gate.lib.carbonpy_dense_container_run.argtypes = [
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_dense_container_run.restype = ctypes.c_int

    gate.lib.carbonpy_block_container_new.argtypes = [
        ctypes.POINTER(gate.handle_type),
        gate.handle_type,
        gate.handle_type,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    gate.lib.carbonpy_block_container_new.restype = ctypes.c_int
    gate.lib.carbonpy_block_container_delete.argtypes = [ctypes.POINTER(gate.handle_type)]
    gate.lib.carbonpy_block_container_delete.restype = ctypes.c_int
    gate.lib.carbonpy_block_container_bind_field.argtypes = [
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_block_container_bind_field.restype = ctypes.c_int
    gate.lib.carbonpy_block_container_run.argtypes = [
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_block_container_run.restype = ctypes.c_int

    gate.lib.carbonpy_mgrid_container_new.argtypes = [
        ctypes.POINTER(gate.handle_type),
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    gate.lib.carbonpy_mgrid_container_new.restype = ctypes.c_int
    gate.lib.carbonpy_mgrid_container_delete.argtypes = [ctypes.POINTER(gate.handle_type)]
    gate.lib.carbonpy_mgrid_container_delete.restype = ctypes.c_int
    gate.lib.carbonpy_mgrid_container_bind_field.argtypes = [
        gate.handle_type,
        gate.handle_type,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_mgrid_container_bind_field.restype = ctypes.c_int
    gate.lib.carbonpy_mgrid_container_run.argtypes = [
        gate.handle_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gate.lib.carbonpy_mgrid_container_run.restype = ctypes.c_int
