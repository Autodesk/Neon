#!/usr/bin/env python3
"""Block container run via CUDA Driver API + Warp CUfunction hooks."""

import ctypes
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CarbonWarpBlockContainerRun(unittest.TestCase):
    def test_block_container_fill(self) -> None:
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
        from carbon.container import BlockContainer, CudaDriver  # noqa: WPS433

        gate = carbon.init(enable_warp=True)
        if gate.lib.carbonpy_device_count() <= 0:
            self.skipTest("no CUDA devices visible")

        system = carbon.System(gate)
        block = carbon.Block(system, 16, 16, 16)
        field = block.new_field(1, dtype=wp.float32)
        driver = CudaDriver(system)

        @wp.func
        def fill_cell(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            iter_idx: int,
            x: int,
            y: int,
            z: int,
        ):
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(span, b, x, y, z)
            wp.carbon.write(
                part, span, cell, 0, wp.float32(b + x + y + z)
            )

        def fill_loader(loader, bind_field):
            bind_field(field)
            loader.declare_kernel(fill_cell)

        container = BlockContainer(driver, block, fill_loader)
        for gpu in range(gate.lib.carbonpy_device_count()):
            container.bind_field(field, gpu)
        container.run(data_view=0)
        wp.synchronize()

        span = block.get_span()
        part = field.get_partition()
        mismatches = wp.zeros(1, dtype=wp.int32, device="cuda:0")

        @wp.kernel
        def check(
            part: gate.CarbonBlockPartition,
            span: gate.CarbonBlockSpan,
            out: wp.array(dtype=wp.int32),
        ):
            iter_idx, x, y, z = wp.tid()
            b = wp.carbon.view_block_idx(span, iter_idx)
            if not wp.carbon.is_active(span, b, x, y, z):
                return
            cell = wp.carbon.set(span, b, x, y, z)
            got = wp.carbon.read(part, span, cell, 0)
            want = wp.float32(b + x + y + z)
            if got != want:
                out[0] = 1

        wp.launch(
            check,
            dim=(int(span.iter_block_count), int(span.mem_x), int(span.mem_y), int(span.mem_z)),
            inputs=[part, span, mismatches],
            device="cuda:0",
        )
        wp.synchronize()
        self.assertEqual(int(mismatches.numpy()[0]), 0)

        container.delete()
        driver.delete()
        field.delete()
        block.delete()
        system.delete()


if __name__ == "__main__":
    unittest.main()
