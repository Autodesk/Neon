# Carbon Python / NVIDIA Warp bindings

CUDA-only Python bindings over `libcarbonpy`. Kernel launches use NVIDIA Warp with
**external native types** (`CarbonDenseSpan`, `CarbonDensePartition`, …) registered through
`wp.types.add_type()`.

## Warp dependency (required fork)

Upstream Warp does not yet ship external-source / external-type support. Carbon vendors the
fork maintained by Lukasz Stolarczyk as a **git submodule** at `warp/`:

```bash
git submodule update --init --recursive warp
./scripts/setup_warp.sh
```

The submodule tracks `https://github.com/nvlukasz/warp.git` on branch
`external-source-support-update2` (see `.gitmodules`). `setup_warp.sh` initializes the
submodule if needed, builds the native library (`python3 build_lib.py`), and `pip install -e`s it.

## libcarbonpy

Build Carbon with Python bindings enabled (CUDA backend only):

```bash
cmake -S . -B build-py -DCARBON_ENABLE_PYTHON=ON
cmake --build build-py -j
```

This produces `py/carbon/lib/linux_<arch>/libcarbonpy.so`.

## Usage sketch

```python
import carbon
import warp as wp

gate = carbon.init(enable_warp=True)

system = carbon.System(gate)
dense = carbon.Dense(system, 8, 8, 8)
field = dense.new_field(components=1, dtype=wp.float32)  # f32/f64/i32/i64
span = dense.get_span()
part = field.get_partition()
```

The object-oriented API mirrors Neon's Warp bindings: `System` owns the multi-GPU
state, grids (`Dense`, `Block`, `Mgrid`) are created from it, and `new_field(dtype=)`
mints scalar-typed field handles (`wp.float32` / `wp.float64` / `wp.int32` / `wp.int64`).
`get_span` / `get_partition` return the blittable POD structs passed straight into Warp
kernels; every object exposes `delete()` and is usable as a context manager.

Registration must happen **before** defining or launching kernels that use Carbon POD types.
External structs are passed by value today; field access inside `@wp.kernel` bodies is not
supported yet — use the `wp.carbon` namespace (`read`, `write`, `is_active`, …).
Same-cell field access is **Idx-only**: ``cell = wp.carbon.set(span, …)`` then
``wp.carbon.read(part, span, cell, component)`` / ``wp.carbon.write(part, span, cell,
component, value)``. Explicit-coordinate ``read`` / ``write`` overloads are not part of
the programming model. Mgrid cross-level child reads use ``wp.carbon.read_child(part, idx,
ox, oy, oz, component)`` (mirrors Neon's ``neon_read_child``).
Each field's ``get_partition()`` returns a dtype-specific partition type so
``wp.carbon.read(part, …)`` resolves the scalar type from the partition argument.
See
`tests/test_warp_external_types.py` and `tests/test_warp_dtypes.py`.

## Dense container (CUDA Driver path)

For Neon-style container execution without `wp.launch`, compile Warp kernels once per
(GPU × DataView), capture `module_exec.get_kernel_hooks(kernel).forward`, and drive them through
Carbon's CUDA Driver API:

```python
from carbon.container import CudaDriver, DenseContainer

driver = CudaDriver(system)

@wp.func
def fill_cell(part: gate.CarbonDensePartition, span: gate.CarbonDenseSpan, ix: int, iy: int, iz: int):
    cell = wp.carbon.set(span, ix, iy, iz)
    wp.carbon.write(part, span, cell, 0, wp.float32(ix + iy + iz))

def fill_loader(loader, bind_field):
    bind_field(field)
    loader.declare_kernel(fill_cell)

container = DenseContainer(driver, dense, fill_loader)
container.bind_field(field, device=0)
container.run(data_view=0)
```

See `tests/test_warp_container_run.py`. Declare `@wp.func` helpers at module/test scope (not
nested inside the loading lambda) so Warp can resolve type annotations.

## Kernel decorator

For the common case — one per-cell function plus field binding — use `@dense.kernel` (or
`carbon.dense_kernel`) instead of hand-writing a loading lambda:

```python
from carbon.container import CudaDriver

driver = CudaDriver(system)

@dense.kernel(driver, fields=[field], access=["write"])
def fill_cell(part, span, ix, iy, iz):
    cell = wp.carbon.set(span, ix, iy, iz)
    wp.carbon.write(part, span, cell, 0, wp.float32(ix + iy + iz))

fill_cell.run(data_view=0)   # already bound on every visible GPU
```

Two-field kernels (e.g. Jacobi) pass `fields=[u_field, v_field]` and optional per-field
`access=["read", "write"]`. The decorated name is a runnable `DenseContainer` (also
composable with `Skeleton.add`). Block and mGrid grids expose `.kernel(...)` with the same
pattern. The explicit `loading_fn` / `declare_kernel` API remains available.

See `tests/test_warp_kernel_decorator.py` and `py/examples/jacobi_step.py`.

## Neon-style loading lambda (`@carbon.kernel`)

For the same idiom as Neon's container factories — a loading lambda that returns
handles via ``loader.get_read_handle`` / ``get_write_handle``, and a compute
``@wp.func`` that captures the handle plus ``loader.get_span()`` and uses
``wp.carbon.read(part, span, cell, c)`` — use ``@carbon.kernel``:

```python
import typing
import carbon

@carbon.kernel  # name defaults to the @wp.func name ("add_kernel")
def add_operator(field):
    def setup(loader: carbon.Loader):
        loader.set_grid(field.get_grid())
        f = loader.get_read_write_handle(field)
        s = loader.get_span()

        @wp.func
        def add_kernel(idx: typing.Any):
            value = wp.carbon.read(f, s, idx, 0)
            g = wp.carbon.global_idx(s, idx)
            extra = wp.float32(
                wp.carbon.get_x(g) + wp.carbon.get_y(g) + wp.carbon.get_z(g)
            )
            wp.carbon.write(f, s, idx, 0, value + extra)

        loader.declare_kernel(add_kernel)

    return setup

op = add_operator(field)
op.run()
```

mGrid kernels call ``loader.set_mres_grid(field.get_grid(), level)`` and
``get_mres_read_handle`` / ``get_mres_write_handle``. Compose sequences with
``Skeleton(system).sequence("name", [add_operator(field)]).run()``. See
``tests/test_warp_kernel_factory.py`` and ``tests/test_warp_skeleton_factory.py``.

## Stencil neighbour reads

Mirror C++ ``Partition::neighbour(span, idx, offset, component)`` with either the
structured return or a folded default:

```python
# NeighbourValue form (matches C++ NeighbourValue<T>)
xp = wp.carbon.neighbour(u, span, idx, 1, 0, 0, 0)
acc = wp.carbon.value_or(xp, wp.float32(0.0))

# Folded convenience (neighbour + value_or in one call)
acc = wp.carbon.read_ngh(u, span, idx, 1, 0, 0, 0, wp.float32(0.0))
```

Dense neighbour checks buffer bounds and global-domain z (``PeriodicZ`` when
``span.boundary_mode == 1``). Block / Mgrid same-level neighbour walks the
27-way connectivity table and active bitmask. See ``py/examples/jacobi_step.py``
and ``tests/test_warp_neighbour.py``.

## Skeleton sequences

`py/carbon/skeleton.py` runs an ordered list of Dense/Block containers through the Driver path.
The API mirrors Neon's skeleton: `Skeleton(system).sequence(name, containers, occ=OCC.none())`
then `run()`:

```python
import carbon

sk = carbon.Skeleton(system)
sk.sequence("pipeline", [fill_container, scale_container])
sk.run()
```

`add(...)` + `compile()` + `run(...)` are still available for finer control.

Halo exchange is inserted automatically: any Dense field pulled with a stencil handle
(`loader.get_read_handle(field, operation=Operation.stencil)`) has its multi-GPU halo
exchanged before the compute pass runs — no manual `make_halo()` / `halo.run()` needed. The
`occ` argument (`carbon.OCC.none()/standard()/extended()/twoWayExtended()`) selects the
`carbon::OverlapMode` overlap strategy; scheduling, auto-halo insertion, and the Internal/
Boundary OCC rewrite are delegated to the **C++ `carbon::Skeleton` engine** (Warp kernels
are foreign-launch `carbon::Kernel` nodes that replay stored `CUfunction` hooks on the
skeleton's `carbon::Stream`s). All four modes are supported with real halo/compute overlap.
See `tests/test_warp_skeleton_halo.py`, `tests/test_warp_skeleton_occ.py`, and
`py/examples/jacobi_step.py` for the multi-GPU Jacobi step.

For direct field-level halo exchange (Neon `field.halo_update_run(stream_idx)`), call
`field.halo_update_run(stream_idx=0)` on a `DenseField`. The halo descriptor is built and
cached on first use; `stream_idx` is accepted for Neon API parity but is not yet wired to
stream selection (issues on `System::stream()` today). See
`tests/test_warp_field_halo_update.py`.

## Wheel packaging

After configuring/building with `-DCARBON_ENABLE_PYTHON=ON`:

```bash
CARBONPY_BUILD_DIR=$PWD/build-py pip install -e py/
```

`py/setup.py` copies `libcarbon.so` and `libcarbonpy.so` into the `carbon` package.

## Examples

See [`examples/README.md`](examples/README.md). Quick start:

```bash
export PYTHONPATH=warp:py
python3 py/examples/jacobi_step.py
python3 py/examples/salt_pepper_filter.py
```
