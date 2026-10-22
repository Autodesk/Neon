#!/usr/bin/env python3
"""Dense fields round-trip across every supported dtype (f32/f64/i32/i64)."""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpDtypes(unittest.TestCase):
    def test_dense_fill_readback_all_dtypes(self) -> None:
        root = _repo_root()
        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")

        sys.path.insert(0, os.path.join(root, "py"))
        try:
            import warp as wp
        except ImportError:
            self.skipTest("warp module not installed")

        repo = getattr(wp, "__file__", "") or ""
        if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
            self.skipTest("Warp external-type fork not installed; run ./scripts/setup_warp.sh")

        import carbon  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        @wp.kernel
        def fill_f32(part: gate.CarbonDensePartition, span: gate.CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(ix + iy + iz)
            )

        @wp.kernel
        def check_f32(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            if wp.carbon.read(part, span, cell, 0) != wp.float32(ix + iy + iz):
                out[0] = 1

        @wp.kernel
        def fill_f64(part: gate.CarbonDensePartitionF64, span: gate.CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.float64(ix + iy + iz)
            )

        @wp.kernel
        def check_f64(
            part: gate.CarbonDensePartitionF64,
            span: gate.CarbonDenseSpan,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            if wp.carbon.read(part, span, cell, 0) != wp.float64(ix + iy + iz):
                out[0] = 1

        @wp.kernel
        def fill_i32(part: gate.CarbonDensePartitionI32, span: gate.CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.int32(ix + iy + iz)
            )

        @wp.kernel
        def check_i32(
            part: gate.CarbonDensePartitionI32,
            span: gate.CarbonDenseSpan,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            if wp.carbon.read(part, span, cell, 0) != wp.int32(ix + iy + iz):
                out[0] = 1

        @wp.kernel
        def fill_i64(part: gate.CarbonDensePartitionI64, span: gate.CarbonDenseSpan):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            wp.carbon.write(
                part, span, cell, 0, wp.int64(ix + iy + iz)
            )

        @wp.kernel
        def check_i64(
            part: gate.CarbonDensePartitionI64,
            span: gate.CarbonDenseSpan,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            if wp.carbon.read(part, span, cell, 0) != wp.int64(ix + iy + iz):
                out[0] = 1

        cases = [
            (wp.float32, fill_f32, check_f32),
            (wp.float64, fill_f64, check_f64),
            (wp.int32, fill_i32, check_i32),
            (wp.int64, fill_i64, check_i64),
        ]

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        span = dense.get_span()
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)

        for dtype, fill_k, check_k in cases:
            field = dense.new_field(1, dtype=dtype)
            self.assertEqual(field.builtin_suffix, {
                wp.float32: "f32",
                wp.float64: "f64",
                wp.int32: "i32",
                wp.int64: "i64",
            }[dtype])
            part = field.get_partition()
            wp.launch(fill_k, dim=(nx, ny, nz), inputs=[part, span], device="cuda:0")
            wp.synchronize()
            mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")
            wp.launch(
                check_k, dim=(nx, ny, nz), inputs=[part, span, mismatches], device="cuda:0"
            )
            wp.synchronize()
            self.assertEqual(
                int(mismatches.numpy()[0]), 0, f"dtype {dtype} round-trip mismatch"
            )
            field.delete()

        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
