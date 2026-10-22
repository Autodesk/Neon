"""Object-oriented MultiResBlock (mGrid) bindings for Carbon + NVIDIA Warp."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from ..dtypes import resolve_dtype

if TYPE_CHECKING:
    from ..system import System


class MgridField:
    """An mGrid field specialized on a scalar dtype and component count."""

    def __init__(self, mgrid: "Mgrid", components: int, dtype) -> None:
        self.mgrid = mgrid
        self.gate = mgrid.gate
        self.components = components
        info = resolve_dtype(dtype)
        self.dtype = dtype
        self.dtype_code = info.code
        self.builtin_suffix = info.suffix
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_mgrid_field_new(
            ctypes.byref(self.handle), mgrid.handle, components, info.code
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_field_new failed")

    def get_grid(self) -> "Mgrid":
        return self.mgrid

    @property
    def num_levels(self) -> int:
        return self.mgrid.depth

    def fill_run(self, level: int, value, stream_idx: int = 0) -> None:
        """Set every component of every active cell at ``level`` to ``value``."""
        del stream_idx
        from .._fieldops import mgrid_fill

        mgrid_fill(self, value, level)

    def zero_run(self, level: int, stream_idx: int = 0) -> None:
        """Zero every component of every active cell at ``level``."""
        self.fill_run(level, 0, stream_idx=stream_idx)

    def copy_from_run(self, level: int, other: "MgridField", stream_idx: int = 0) -> None:
        """Copy every active cell at ``level`` from ``other`` into this field."""
        del stream_idx
        from .._fieldops import mgrid_copy

        mgrid_copy(self, other, level)

    def to_numpy(self, level: int):
        """Gather every active cell at ``level`` from all devices into a host array.

        Returns an array of shape ``(components, gx, gy, gz)`` where ``(gx, gy, gz)``
        is the level's global voxel extent. Inactive cells read back as zero.
        """
        from .._fieldops import mgrid_to_numpy

        return mgrid_to_numpy(self, level)

    def update_host(self, level: int, stream_idx: int = 0):
        """Snapshot a level's device storage into a host mirror (Neon parity)."""
        del stream_idx
        mirror = self.to_numpy(level)
        if not hasattr(self, "_host_mirror"):
            self._host_mirror = {}
        self._host_mirror[level] = mirror
        return mirror

    def export_vti(
        self,
        filename: str,
        level: int,
        field_name: str = "field",
        stream_idx: int = 0,
    ) -> None:
        """Write a level to a VTK ImageData (``.vti``) file (Neon parity).

        Uses the host mirror from the most recent :meth:`update_host` for that
        level; if none exists it is gathered on demand.
        """
        import numpy as np
        import warp as wp

        mirror = getattr(self, "_host_mirror", {}).get(level)
        if mirror is None:
            mirror = self.update_host(level, stream_idx)

        c, nx, ny, nz = mirror.shape
        vtk_dtype = {
            wp.float32: "Float32",
            wp.float64: "Float64",
            wp.int32: "Int32",
            wp.int64: "Int64",
            wp.uint8: "UInt8",
        }.get(self.dtype, "Float32")

        # VTK point order is x-fastest then y then z, with components innermost.
        flat = np.transpose(mirror, (3, 2, 1, 0)).reshape(-1)
        values = " ".join(repr(v.item()) for v in flat)

        extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
        xml = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
            f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">\n'
            f'    <Piece Extent="{extent}">\n'
            "      <PointData>\n"
            f'        <DataArray type="{vtk_dtype}" Name="{field_name}" '
            f'NumberOfComponents="{c}" format="ascii">\n'
            f"          {values}\n"
            "        </DataArray>\n"
            "      </PointData>\n"
            "    </Piece>\n"
            "  </ImageData>\n"
            "</VTKFile>\n"
        )
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(xml)

    def get_partition(self, level: int, device: int = 0, data_view: int = 0):
        from ..warp_types import BLOCK_PARTITION_BY_SUFFIX

        part = BLOCK_PARTITION_BY_SUFFIX[self.builtin_suffix]()
        rc = self.gate.lib.carbonpy_mgrid_field_get_partition(
            self.handle, level, device, data_view, ctypes.byref(part)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_field_get_partition failed")
        return part

    def get_cross_partition(self, level: int, device: int = 0, data_view: int = 0):
        from ..warp_types import MGRID_PARTITION_BY_SUFFIX

        part = MGRID_PARTITION_BY_SUFFIX[self.builtin_suffix]()
        rc = self.gate.lib.carbonpy_mgrid_field_get_cross_partition(
            self.handle, level, device, data_view, ctypes.byref(part)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_field_get_cross_partition failed")
        return part

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_mgrid_field_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "MgridField":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


class Mgrid:
    """A multi-resolution block octree grid over a :class:`System`."""

    name = "Mgrid"

    def __init__(
        self,
        system: "System",
        finest_x: int,
        finest_y: int,
        finest_z: int,
        depth: int = 2,
        ref_factor: int = 2,
        sparsity_pattern_list=None,
        sparsity_pattern_origins=None,
    ) -> None:
        """Build a multi-resolution octree over ``system``.

        With no ``sparsity_pattern_list`` the grid is **fully active** (every voxel of every
        level retained). Passing per-level dense masks + origins builds a *sparse* grid,
        mirroring Neon's pyNeon ``mGrid(sparsity_pattern_list, sparsity_pattern_origins)``:

        - ``sparsity_pattern_list``: one array per level in level order (index 0 = finest),
          each a dense int32/bool array of shape ``(sx, sy, sz)`` in **level-local** voxel
          coordinates; non-zero == active. ``depth`` is taken from the list length.
        - ``sparsity_pattern_origins``: one ``(x, y, z)`` (tuple or ``Index_3d``) per level, the
          level-local offset where that mask's ``(0, 0, 0)`` sits. Defaults to ``(0, 0, 0)``.
        """
        self.system = system
        self.gate = system.gate
        self.finest = (int(finest_x), int(finest_y), int(finest_z))
        self.ref_factor = int(ref_factor)
        self.handle = self.gate.handle_type()

        if sparsity_pattern_list is not None:
            self._init_sparse(system, sparsity_pattern_list, sparsity_pattern_origins)
            return

        rc = self.gate.lib.carbonpy_mgrid_new(
            ctypes.byref(self.handle),
            system.handle,
            ctypes.c_int64(finest_x),
            ctypes.c_int64(finest_y),
            ctypes.c_int64(finest_z),
            depth,
            ref_factor,
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_new failed")

    def _init_sparse(self, system, masks, origins) -> None:
        import numpy as np

        depth = len(masks)
        if depth <= 0:
            raise ValueError("Mgrid: sparsity_pattern_list must have at least one level")
        if origins is None:
            origins = [(0, 0, 0)] * depth
        if len(origins) != depth:
            raise ValueError("Mgrid: sparsity_pattern_origins length must match sparsity_pattern_list")

        shapes = np.zeros((depth, 3), dtype=np.int64)
        origins_arr = np.zeros((depth, 3), dtype=np.int64)
        flat_parts = []
        for lvl, mask in enumerate(masks):
            m = np.ascontiguousarray(mask, dtype=np.int32)
            if m.ndim != 3:
                raise ValueError(f"Mgrid: level {lvl} mask must be 3-D, got shape {m.shape}")
            shapes[lvl] = m.shape
            o = origins[lvl]
            origins_arr[lvl] = (
                (int(o.x), int(o.y), int(o.z)) if hasattr(o, "x") else tuple(int(v) for v in o)
            )
            flat_parts.append(m.reshape(-1))

        flat = (
            np.concatenate(flat_parts).astype(np.int32, copy=False)
            if flat_parts
            else np.zeros(0, dtype=np.int32)
        )
        # Keep buffers alive for the duration of the C call.
        shapes = np.ascontiguousarray(shapes)
        origins_arr = np.ascontiguousarray(origins_arr)
        flat = np.ascontiguousarray(flat)

        masks_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if flat.size else None
        rc = self.gate.lib.carbonpy_mgrid_new_sparse(
            ctypes.byref(self.handle),
            system.handle,
            ctypes.c_int64(self.finest[0]),
            ctypes.c_int64(self.finest[1]),
            ctypes.c_int64(self.finest[2]),
            depth,
            self.ref_factor,
            masks_ptr,
            shapes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            origins_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_new_sparse failed")

    @property
    def depth(self) -> int:
        depth = self.gate.lib.carbonpy_mgrid_depth(self.handle)
        if depth < 0:
            raise RuntimeError("carbonpy_mgrid_depth failed")
        return depth

    def level_extent(self, level: int) -> tuple:
        """Global (x, y, z) voxel extent of ``level`` (finest is level 0)."""
        r = self.ref_factor**level
        fx, fy, fz = self.finest
        return (fx // r, fy // r, fz // r)

    def new_field(self, components: int = 1, dtype=None) -> MgridField:
        if dtype is None:
            import warp as wp

            dtype = wp.float32
        return MgridField(self, components, dtype)

    def get_span(self, level: int, device: int = 0, data_view: int = 0):
        span = self.gate.CarbonBlockSpan()
        rc = self.gate.lib.carbonpy_mgrid_get_span(
            self.handle, level, device, data_view, ctypes.byref(span)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_mgrid_get_span failed")
        return span

    def kernel(
        self,
        driver,
        level: int,
        field,
        *,
        name: str = "carbon_kernel",
        access=None,
    ):
        from ..kernel import mgrid_kernel

        return mgrid_kernel(driver, self, level, field, name=name, access=access)

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_mgrid_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "Mgrid":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()
