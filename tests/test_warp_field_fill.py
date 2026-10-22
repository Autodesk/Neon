#!/usr/bin/env python3
"""DenseField.zero_run / fill_run (Neon-parity field initialization)."""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpFieldFill(unittest.TestCase):
    def _setup(self):
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
        return wp, carbon, gate

    def _count_mismatches(self, wp, gate, dense, field, want, ncomp: int) -> int:
        span = dense.get_span()
        part = field.get_partition()
        nx, ny, nz = int(span.iter_x), int(span.iter_y), int(span.iter_z)
        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(
            part: gate.CarbonDensePartition,
            span: gate.CarbonDenseSpan,
            want: wp.float32,
            ncomp: int,
            out: wp.array(dtype=wp.int32),
        ):
            ix, iy, iz = wp.tid()
            cell = wp.carbon.set(span, ix, iy, iz)
            for c in range(ncomp):
                got = wp.carbon.read(part, span, cell, c)
                if got != want:
                    out[0] = 1

        wp.launch(
            check,
            dim=(nx, ny, nz),
            inputs=[part, span, wp.float32(want), ncomp, mismatches],
            device="cuda:0",
        )
        wp.synchronize()
        return int(mismatches.numpy()[0])

    def test_zero_and_fill(self) -> None:
        wp, carbon, gate = self._setup()

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        field = dense.new_field(3, dtype=wp.float32)

        field.fill_run(2.5)
        self.assertEqual(self._count_mismatches(wp, gate, dense, field, 2.5, 3), 0)

        field.zero_run()
        self.assertEqual(self._count_mismatches(wp, gate, dense, field, 0.0, 3), 0)

        field.delete()
        dense.delete()
        system.delete()

    def test_copy_from_run(self) -> None:
        wp, carbon, gate = self._setup()

        system = carbon.System(gate)
        dense = carbon.Dense(system, 8, 8, 8)
        src = dense.new_field(3, dtype=wp.float32)
        dst = dense.new_field(3, dtype=wp.float32)

        src.fill_run(3.25)
        dst.zero_run()
        dst.copy_from_run(src)
        self.assertEqual(self._count_mismatches(wp, gate, dense, dst, 3.25, 3), 0)

        dst.delete()
        src.delete()
        dense.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
