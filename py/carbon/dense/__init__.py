"""Object-oriented Dense grid bindings for Carbon + NVIDIA Warp.

Mirrors Neon's Warp API shape: :class:`Dense` owns the grid, ``new_field`` mints
typed :class:`DenseField` handles, and ``get_span`` / ``get_partition`` return the
blittable POD structs passed straight into Warp kernels.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from ..dtypes import resolve_dtype

if TYPE_CHECKING:
    from ..gate import Gate
    from ..system import System


class DenseHalo:
    """Intra-process halo exchange for a :class:`DenseField`."""

    def __init__(self, field: "DenseField") -> None:
        self.field = field
        self.system = field.dense.system
        self.gate = field.gate
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_dense_halo_new(
            ctypes.byref(self.handle), field.handle
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_halo_new failed")

    def run(self) -> None:
        rc = self.gate.lib.carbonpy_dense_halo_run(
            self.handle, self.system.handle
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_halo_run failed")

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_dense_halo_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "DenseHalo":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


class DenseField:
    """A Dense field specialized on a scalar dtype and component count."""

    def __init__(self, dense: "Dense", components: int, dtype) -> None:
        self.dense = dense
        self.gate = dense.gate
        self.components = components
        info = resolve_dtype(dtype)
        self.dtype = dtype
        self.dtype_code = info.code
        self.builtin_suffix = info.suffix
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_dense_field_new(
            ctypes.byref(self.handle), dense.handle, components, info.code
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_field_new failed")
        self._halo: DenseHalo | None = None

    def get_grid(self) -> "Dense":
        return self.dense

    def get_partition(self, device: int = 0, data_view: int = 0):
        from ..warp_types import DENSE_PARTITION_BY_SUFFIX

        part = DENSE_PARTITION_BY_SUFFIX[self.builtin_suffix]()
        rc = self.gate.lib.carbonpy_dense_field_get_partition(
            self.handle, device, data_view, ctypes.byref(part)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_field_get_partition failed")
        return part

    def fill_run(self, value, stream_idx: int = 0) -> None:
        """Set every component of every owned cell to ``value`` (Neon parity)."""
        del stream_idx
        from .._fieldops import dense_fill

        dense_fill(self, value)

    def zero_run(self, stream_idx: int = 0) -> None:
        """Zero every component of every owned cell (Neon parity)."""
        self.fill_run(0, stream_idx=stream_idx)

    def copy_from_run(self, other, stream_idx: int = 0) -> None:
        """Copy every owned cell from ``other`` into this field (Neon parity)."""
        del stream_idx
        from .._fieldops import dense_copy

        dense_copy(self, other)

    def to_numpy(self, data_view: int = 0):
        """Gather every owned cell from all devices into a host ``numpy`` array.

        Returns an array of shape ``(components, global_x, global_y, global_z)``
        (component-major, matching XLB's WARP field convention). Each per-device
        slab is read back independently by wrapping its partition storage pointer
        as a non-owning Warp array on that device and copying to host, so this is
        correct across multiple GPUs (no cross-device access).
        """
        import numpy as np
        import warp as wp

        gx = self.dense.dim.x
        gy = self.dense.dim.y
        gz = self.dense.dim.z
        c = self.components
        out = np.zeros((c, gx, gy, gz), dtype=wp.dtype_to_numpy(self.dtype))

        n_dev = self.dense.system.get_num_devices()
        for dev in range(n_dev):
            part = self.get_partition(dev, data_view)
            span = part.span
            halo_z = int(span.halo_z)
            logic_x = int(span.logic_x)
            logic_y = int(span.logic_y)
            logic_z = int(span.logic_z)
            pitch_xy = int(span.pitch_xy)
            haloed_z = logic_z + 2 * halo_z
            n_elem = c * haloed_z * pitch_xy
            if n_elem == 0:
                continue

            # Alias the partition storage (no deleter -> Warp will not free it).
            wp_buf = wp.array(
                ptr=int(part.data_ptr),
                dtype=self.dtype,
                shape=(n_elem,),
                device=f"cuda:{dev}",
            )
            host = wp_buf.numpy().reshape(c, haloed_z, pitch_xy)
            # Drop xy padding, split the packed xy slab into (y, x), and keep only the
            # owned z rows (buffer rows [halo_z, halo_z + logic_z)).
            host = host[:, :, : logic_x * logic_y].reshape(c, haloed_z, logic_y, logic_x)
            owned = host[:, halo_z : halo_z + logic_z, :, :]  # (c, z, y, x)
            owned = np.transpose(owned, (0, 3, 2, 1))  # (c, x, y, z)

            gz0 = halo_z + int(span.z_offset)  # global z of this slab's owned iz == 0
            out[:, :, :, gz0 : gz0 + logic_z] = owned

        return out

    def update_host(self, stream_idx: int = 0):
        """Snapshot device storage into a host mirror (Neon parity).

        Stores the gathered array on the field; :meth:`export_vti` reuses it.
        """
        del stream_idx
        self._host_mirror = self.to_numpy()
        return self._host_mirror

    def export_vti(self, filename: str, field_name: str = "field", stream_idx: int = 0):
        """Write the field to a VTK ImageData (``.vti``) file (Neon parity).

        Uses the host mirror from the most recent :meth:`update_host`; if none
        exists it is gathered on demand.
        """
        import numpy as np
        import warp as wp

        mirror = getattr(self, "_host_mirror", None)
        if mirror is None:
            mirror = self.update_host(stream_idx)

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

    def make_halo(self) -> DenseHalo:
        return DenseHalo(self)

    def halo_update_run(self, stream_idx: int = 0) -> None:
        del stream_idx
        if self._halo is None:
            self._halo = self.make_halo()
        self._halo.run()

    def delete(self) -> None:
        if self._halo is not None:
            self._halo.delete()
            self._halo = None
        if self.handle.value:
            self.gate.lib.carbonpy_dense_field_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "DenseField":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()


class Dense:
    """A dense Cartesian z-slab grid over a :class:`System`."""

    name = "Dense"

    def __init__(
        self,
        system: "System",
        extent_x: int,
        extent_y: int,
        extent_z: int,
        *,
        halo_radius: int = 0,
    ) -> None:
        self.system = system
        self.gate = system.gate
        from ..index3d import Index_3d

        self.dim = Index_3d(extent_x, extent_y, extent_z)
        self.handle = self.gate.handle_type()
        rc = self.gate.lib.carbonpy_dense_new(
            ctypes.byref(self.handle),
            system.handle,
            ctypes.c_int64(extent_x),
            ctypes.c_int64(extent_y),
            ctypes.c_int64(extent_z),
            ctypes.c_int(halo_radius),
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_new failed")

    @property
    def backend(self):
        """Neon-parity accessor: the owning :class:`System` (Neon's ``grid.backend``)."""
        return self.system

    def new_field(self, components: int = 1, dtype=None, *, cardinality: int | None = None) -> DenseField:
        if cardinality is not None:
            components = cardinality
        if dtype is None:
            import warp as wp

            dtype = wp.float32
        return DenseField(self, components, dtype)

    def get_span(self, device: int = 0, data_view: int = 0):
        span = self.gate.CarbonDenseSpan()
        rc = self.gate.lib.carbonpy_dense_get_span(
            self.handle, device, data_view, ctypes.byref(span)
        )
        if rc != 0:
            raise RuntimeError("carbonpy_dense_get_span failed")
        return span

    def kernel(
        self,
        driver,
        fields,
        *,
        name: str = "carbon_kernel",
        access=None,
    ):
        from ..kernel import dense_kernel

        return dense_kernel(driver, self, fields, name=name, access=access)

    def delete(self) -> None:
        if self.handle.value:
            self.gate.lib.carbonpy_dense_delete(ctypes.byref(self.handle))

    def __enter__(self) -> "Dense":
        return self

    def __exit__(self, *exc) -> None:
        self.delete()
