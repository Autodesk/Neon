#!/usr/bin/env python3
"""Multi-GPU Jacobi under all Skeleton OCC modes via the C++ carbon::Skeleton binding."""

import os
import sys
import typing
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jacobi_reference(u):
    import numpy as np

    nx, ny, nz = u.shape
    v = np.zeros_like(u)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                acc = 0.0
                if ix > 0:
                    acc += u[ix - 1, iy, iz]
                if ix + 1 < nx:
                    acc += u[ix + 1, iy, iz]
                if iy > 0:
                    acc += u[ix, iy - 1, iz]
                if iy + 1 < ny:
                    acc += u[ix, iy + 1, iz]
                if iz > 0:
                    acc += u[ix, iy, iz - 1]
                if iz + 1 < nz:
                    acc += u[ix, iy, iz + 1]
                v[ix, iy, iz] = acc / 6.0
    return v


class CarbonWarpSkeletonOcc(unittest.TestCase):
    def _run_jacobi_once(self, occ) -> tuple[float, int]:
        import numpy as np
        import warp as wp

        import carbon
        from carbon.loader import Operation
        from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan

        global_x = global_y = global_z = 8
        system = carbon.System(self.gate)
        dense = carbon.Dense(system, global_x, global_y, global_z, halo_radius=1)
        u_field = dense.new_field(1, dtype=wp.float32)
        v_field = dense.new_field(1, dtype=wp.float32)
        n_gpus = self.gate.lib.carbonpy_device_count()

        @wp.kernel
        def seed(part: CarbonDensePartition, span: CarbonDenseSpan, z_origin: int):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(part, span, cell, 0, wp.float32(ix + iy + z_origin + iz))

        @wp.kernel
        def copy_from_carbon(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            out: wp.array(dtype=wp.float32, ndim=3),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            out[ix, iy, iz] = wp.carbon.read(part, span, cell, 0)

        for gpu in range(n_gpus):
            u_part = u_field.get_partition(gpu)
            u_span = dense.get_span(gpu)
            iter_x, iter_y, iter_z = int(u_span.iter_x), int(u_span.iter_y), int(u_span.iter_z)
            z_origin = int(u_span.z_offset + u_span.halo_z)
            wp.launch(
                seed,
                dim=(iter_x, iter_y, iter_z),
                inputs=[u_part, u_span, z_origin],
                device=f"cuda:{gpu}",
            )
        wp.synchronize()

        @carbon.kernel
        def jacobi_step(u_field, v_field):
            def setup(loader: carbon.Loader):
                loader.set_grid(u_field.get_grid())
                u = loader.get_read_handle(u_field, operation=Operation.stencil)
                v = loader.get_write_handle(v_field)
                span = loader.get_span()

                @wp.func
                def jacobi_kernel(idx: typing.Any):
                    acc = wp.carbon.read_ngh(u, span, idx, -1, 0, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 1, 0, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, -1, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 1, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 0, -1, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 0, 1, 0, wp.float32(0.0))
                    wp.carbon.write(v, span, idx, 0, acc / wp.float32(6.0))

                loader.declare_kernel(jacobi_kernel)

            return setup

        jacobi = jacobi_step(u_field, v_field)
        skel = carbon.Skeleton(system)
        skel.sequence("jacobi", [jacobi], occ=occ)
        halo_count = skel.inserted_halo_count()
        skel.run()
        wp.synchronize()

        got = np.zeros((global_x, global_y, global_z), dtype=np.float32)
        u_host = np.zeros((global_x, global_y, global_z), dtype=np.float32)
        for gpu in range(n_gpus):
            v_span = dense.get_span(gpu)
            u_span = dense.get_span(gpu)
            iter_x, iter_y, iter_z = int(v_span.iter_x), int(v_span.iter_y), int(v_span.iter_z)
            z_off = int(v_span.z_offset + v_span.halo_z)

            slab = wp.zeros((iter_x, iter_y, iter_z), dtype=wp.float32, device=f"cuda:{gpu}")
            wp.launch(
                copy_from_carbon,
                dim=(iter_x, iter_y, iter_z),
                inputs=[v_field.get_partition(gpu), v_span, slab],
                device=f"cuda:{gpu}",
            )
            got[:, :, z_off : z_off + iter_z] = slab.numpy()

            u_slab = wp.zeros((iter_x, iter_y, iter_z), dtype=wp.float32, device=f"cuda:{gpu}")
            wp.launch(
                copy_from_carbon,
                dim=(iter_x, iter_y, iter_z),
                inputs=[u_field.get_partition(gpu), u_span, u_slab],
                device=f"cuda:{gpu}",
            )
            u_host[:, :, z_off : z_off + iter_z] = u_slab.numpy()
        wp.synchronize()

        ref = _jacobi_reference(u_host)
        diff = float(np.abs(got - ref).max())

        skel.delete()
        jacobi.delete()
        u_field.delete()
        v_field.delete()
        dense.delete()
        system.delete()
        return diff, halo_count

    def setUp(self) -> None:
        root = _repo_root()
        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp  # noqa: F401
        except ImportError:
            self.skipTest("warp module not installed")

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        try:
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

        import carbon

        self.gate = carbon.init(enable_warp=True)
        if self.gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

    def test_all_occ_modes_match_reference(self) -> None:
        import carbon

        occ_modes = [
            carbon.SkeletonConfig.OCC.none(),
            carbon.SkeletonConfig.OCC.standard(),
            carbon.SkeletonConfig.OCC.extended(),
            carbon.SkeletonConfig.OCC.twoWayExtended(),
        ]
        for occ in occ_modes:
            with self.subTest(occ=occ.to_string()):
                diff, halo_count = self._run_jacobi_once(occ)
                self.assertEqual(halo_count, 1)
                self.assertLess(diff, 1e-5)

    def test_repeat_iterations_stable(self) -> None:
        import numpy as np
        import warp as wp

        import carbon
        from carbon.loader import Operation
        from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan

        global_x = global_y = global_z = 8
        system = carbon.System(self.gate)
        dense = carbon.Dense(system, global_x, global_y, global_z, halo_radius=1)
        u_field = dense.new_field(1, dtype=wp.float32)
        v_field = dense.new_field(1, dtype=wp.float32)
        n_gpus = self.gate.lib.carbonpy_device_count()

        @wp.kernel
        def seed(part: CarbonDensePartition, span: CarbonDenseSpan, z_origin: int):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(part, span, cell, 0, wp.float32(ix + iy + z_origin + iz))

        @wp.kernel
        def copy_from_carbon(
            part: CarbonDensePartition,
            span: CarbonDenseSpan,
            out: wp.array(dtype=wp.float32, ndim=3),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            out[ix, iy, iz] = wp.carbon.read(part, span, cell, 0)

        for gpu in range(n_gpus):
            u_span = dense.get_span(gpu)
            iter_x, iter_y, iter_z = int(u_span.iter_x), int(u_span.iter_y), int(u_span.iter_z)
            z_origin = int(u_span.z_offset + u_span.halo_z)
            wp.launch(
                seed,
                dim=(iter_x, iter_y, iter_z),
                inputs=[u_field.get_partition(gpu), u_span, z_origin],
                device=f"cuda:{gpu}",
            )
        wp.synchronize()

        @carbon.kernel
        def jacobi_step(u_field, v_field):
            def setup(loader: carbon.Loader):
                loader.set_grid(u_field.get_grid())
                u = loader.get_read_handle(u_field, operation=Operation.stencil)
                v = loader.get_write_handle(v_field)
                span = loader.get_span()

                @wp.func
                def jacobi_kernel(idx: typing.Any):
                    acc = wp.carbon.read_ngh(u, span, idx, -1, 0, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 1, 0, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, -1, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 1, 0, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 0, -1, 0, wp.float32(0.0))
                    acc = acc + wp.carbon.read_ngh(u, span, idx, 0, 0, 1, 0, wp.float32(0.0))
                    wp.carbon.write(v, span, idx, 0, acc / wp.float32(6.0))

                loader.declare_kernel(jacobi_kernel)

            return setup

        jacobi = jacobi_step(u_field, v_field)
        skel = carbon.Skeleton(system)
        skel.sequence("jacobi", [jacobi], occ=carbon.SkeletonConfig.OCC.standard())

        def read_v_global() -> np.ndarray:
            got = np.zeros((global_x, global_y, global_z), dtype=np.float32)
            for gpu in range(n_gpus):
                v_span = dense.get_span(gpu)
                iter_x, iter_y, iter_z = int(v_span.iter_x), int(v_span.iter_y), int(v_span.iter_z)
                z_off = int(v_span.z_offset + v_span.halo_z)
                slab = wp.zeros((iter_x, iter_y, iter_z), dtype=wp.float32, device=f"cuda:{gpu}")
                wp.launch(
                    copy_from_carbon,
                    dim=(iter_x, iter_y, iter_z),
                    inputs=[v_field.get_partition(gpu), v_span, slab],
                    device=f"cuda:{gpu}",
                )
                got[:, :, z_off : z_off + iter_z] = slab.numpy()
            wp.synchronize()
            return got

        snapshots = []
        for _ in range(3):
            skel.run()
            wp.synchronize()
            snapshots.append(read_v_global())

        skel.delete()
        jacobi.delete()
        u_field.delete()
        v_field.delete()
        dense.delete()
        system.delete()

        self.assertTrue(np.allclose(snapshots[0], snapshots[1], atol=1e-5))
        self.assertTrue(np.allclose(snapshots[1], snapshots[2], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
