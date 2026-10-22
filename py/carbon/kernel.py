"""Decorator factories for Carbon Warp kernels."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .loader import AccessMode, AccessPattern, Loader, Operation


def _stencil_dense_fields(tokens) -> list:
    """Dense fields read with a Stencil access pattern (dedup, add-order)."""
    from .dense import DenseField

    seen: set[int] = set()
    fields = []
    for token in tokens:
        if token.pattern != AccessPattern.Stencil:
            continue
        if not isinstance(token.field, DenseField):
            continue
        key = id(token.field)
        if key in seen:
            continue
        seen.add(key)
        fields.append(token.field)
    return fields

_ACCESS_MODES = frozenset({"read", "write", "read_write", "read_stencil"})


def _normalize_dense_access(
    access: Optional[Sequence[str]],
    field_count: int,
) -> List[str]:
    if access is None:
        return ["read_write"] * field_count
    access_list = list(access)
    if len(access_list) != field_count:
        raise ValueError(
            f"access must have {field_count} entries (one per field), got {len(access_list)}"
        )
    for mode in access_list:
        if mode not in _ACCESS_MODES:
            raise ValueError(f"unsupported access mode: {mode!r}")
    return access_list


def _normalize_single_access(access: Optional[str]) -> str:
    if access is None:
        return "read_write"
    if access not in _ACCESS_MODES:
        raise ValueError(f"unsupported access mode: {access!r}")
    return access


def _loader_access(loader, field, access: str) -> None:
    if access == "read":
        loader.read(field)
    elif access == "write":
        loader.write(field)
    elif access == "read_write":
        loader.read_write(field)
    elif access == "read_stencil":
        loader.read_stencil(field)
    else:
        raise ValueError(f"unsupported access mode: {access!r}")


def _wrap_loading_lambda(user_setup: Callable[[Loader], None]) -> Callable:
    def loading_fn(loader: Loader, bind_field) -> None:
        user_setup(loader)

    return loading_fn


def _unique_fields(tokens) -> list:
    seen: set[int] = set()
    fields = []
    for token in tokens:
        key = id(token.field)
        if key in seen:
            continue
        seen.add(key)
        fields.append(token.field)
    return fields


def _default_kernel_name(parser: Loader, fallback: str) -> str:
    if parser.kernels:
        compute = parser.kernels[0]
        key = getattr(compute, "key", None)
        if key:
            return str(key)
        name = getattr(compute, "__name__", None)
        if name:
            return str(name)
    return fallback


class BoundKernel:
    """A compiled Carbon kernel built from a Neon-style loading lambda."""

    def __init__(self, container, driver, stencil_fields=None) -> None:
        self.container = container
        self.driver = driver
        self.name = container.name
        self.stencil_fields = list(stencil_fields) if stencil_fields else []
        self._owns_driver = True

    def run(self, stream_idx: int = 0, data_view: int = 0) -> None:
        self.container.run(stream_idx=stream_idx, data_view=data_view)

    def delete(self) -> None:
        self.container.delete()
        if self._owns_driver:
            self.driver.delete()

    def __enter__(self) -> "BoundKernel":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


def build_bound_kernel(
    loading_lambda: Callable[[Loader], None],
    *,
    name: str | None = None,
    driver=None,
) -> BoundKernel:
    from .block import Block
    from .container import BlockContainer, CudaDriver, DenseContainer, MgridContainer
    from .dense import Dense
    from .multires import Mgrid

    parser = Loader(parsing=True)
    loading_fn = _wrap_loading_lambda(loading_lambda)
    loading_fn(parser, lambda *_args: None)

    grid = parser._retrieve_grid()
    if grid is None:
        raise RuntimeError("loading_fn must call loader.set_grid() or loader.set_mres_grid()")

    kernel_name = name or _default_kernel_name(parser, "carbon_kernel")
    fields = _unique_fields(parser.tokens)
    system = grid.system
    owns_driver = driver is None
    if driver is None:
        driver = CudaDriver(system)
    n_gpus = driver.gate.lib.carbonpy_device_count()

    if isinstance(grid, Dense):
        container = DenseContainer(
            driver,
            grid,
            loading_fn,
            name=kernel_name,
            field_count=max(1, len(fields)),
            capture=True,
        )
        for gpu in range(n_gpus):
            for slot, field in enumerate(fields):
                container.bind_field(field, gpu, slot)
    elif isinstance(grid, Block):
        container = BlockContainer(
            driver,
            grid,
            loading_fn,
            name=kernel_name,
            capture=True,
        )
        for gpu in range(n_gpus):
            for field in fields:
                container.bind_field(field, gpu)
    elif isinstance(grid, Mgrid):
        level = parser.get_mres_level()
        if level is None:
            raise RuntimeError("loading_fn must call loader.set_mres_grid() for Mgrid kernels")
        container = MgridContainer(
            driver,
            grid,
            level,
            loading_fn,
            name=kernel_name,
            capture=True,
        )
        for gpu in range(n_gpus):
            for field in fields:
                container.bind_field(field, gpu)
    else:
        if owns_driver:
            driver.delete()
        raise TypeError(f"unsupported grid type: {type(grid)!r}")

    stencil_fields = _stencil_dense_fields(parser.tokens)
    container.stencil_fields = stencil_fields
    bound = BoundKernel(container, driver, stencil_fields)
    bound._owns_driver = owns_driver
    return bound


def kernel(name_or_func=None, *, name: str | None = None):
    """Decorator to build a bound Carbon kernel from a Neon-style loading lambda."""

    def factory_decorator(loading_lambda_generator: Callable[..., Callable[[Loader], None]]):
        fallback = loading_lambda_generator.__name__

        def kernel_generator(*args, **kwargs):
            loading_lambda = loading_lambda_generator(*args, **kwargs)
            effective_name = name
            if effective_name is None and isinstance(name_or_func, str):
                effective_name = name_or_func
            if effective_name is None:
                effective_name = None
            return build_bound_kernel(
                loading_lambda,
                name=effective_name or None,
            )

        kernel_generator.__name__ = loading_lambda_generator.__name__
        kernel_generator.__doc__ = loading_lambda_generator.__doc__
        return kernel_generator

    if callable(name_or_func):
        return factory_decorator(name_or_func)
    return factory_decorator


def dense_kernel(
    driver,
    grid,
    fields,
    *,
    name: str = "carbon_kernel",
    access: Optional[Sequence[str]] = None,
):
    """Return a decorator that builds a bound :class:`DenseContainer` from a per-cell func."""

    field_list = list(fields)
    field_count = len(field_list)
    if field_count not in (1, 2):
        raise ValueError(f"dense_kernel supports 1 or 2 fields, got {field_count}")
    access_list = _normalize_dense_access(access, field_count)

    def decorator(fn: Callable[..., Any]):
        import warp as wp

        from .container import DenseContainer

        wp_fn = wp.func(fn)

        def loading_fn(loader, bind_field):
            for slot, field in enumerate(field_list):
                bind_field(field, slot)
                _loader_access(loader, field, access_list[slot])
            loader.declare_kernel(wp_fn)

        container = DenseContainer(
            driver,
            grid,
            loading_fn,
            name=name,
            field_count=field_count,
        )
        n_gpus = driver.gate.lib.carbonpy_device_count()
        for gpu in range(n_gpus):
            for slot, field in enumerate(field_list):
                container.bind_field(field, gpu, slot)
        return container

    return decorator


def block_kernel(
    driver,
    grid,
    field,
    *,
    name: str = "carbon_kernel",
    access: Optional[str] = None,
):
    """Return a decorator that builds a bound :class:`BlockContainer` from a per-cell func."""

    access_mode = _normalize_single_access(access)

    def decorator(fn: Callable[..., Any]):
        import warp as wp

        from .container import BlockContainer

        wp_fn = wp.func(fn)

        def loading_fn(loader, bind_field):
            bind_field(field)
            _loader_access(loader, field, access_mode)
            loader.declare_kernel(wp_fn)

        container = BlockContainer(driver, grid, loading_fn, name=name)
        n_gpus = driver.gate.lib.carbonpy_device_count()
        for gpu in range(n_gpus):
            container.bind_field(field, gpu)
        return container

    return decorator


def mgrid_kernel(
    driver,
    grid,
    level: int,
    field,
    *,
    name: str = "carbon_kernel",
    access: Optional[str] = None,
):
    """Return a decorator that builds a bound :class:`MgridContainer` from a per-cell func."""

    access_mode = _normalize_single_access(access)

    def decorator(fn: Callable[..., Any]):
        import warp as wp

        from .container import MgridContainer

        wp_fn = wp.func(fn)

        def loading_fn(loader, bind_field):
            bind_field(field)
            _loader_access(loader, field, access_mode)
            loader.declare_kernel(wp_fn)

        container = MgridContainer(driver, grid, level, loading_fn, name=name)
        n_gpus = driver.gate.lib.carbonpy_device_count()
        for gpu in range(n_gpus):
            container.bind_field(field, gpu)
        return container

    return decorator
