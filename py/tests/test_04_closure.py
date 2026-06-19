import os
import unittest

from env_setup import update_pythonpath

update_pythonpath()

import warp as wp
import neon
from neon import DataView, Index_3d
from neon.dense import dSpan
from neon.dense.dPartition import dPartition_int32

from neon_test_utils import gpu_available, init_warp_neon, require_gpu, require_wpne, setup_wpne_build

try:
    import wpne
    HAS_WPNE = True
except ImportError:
    HAS_WPNE = False


@require_gpu
@require_wpne
class TestClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_warp_neon(verbose=False)
        setup_wpne_build(os.path.dirname(os.path.abspath(__file__)))

    def test_kernel_without_closure(self):
        @wp.kernel
        def kernel():
            wp.neon_print(wp.NeonDenseIdx_create(11, 22, 33))

        with wp.ScopedDevice("cuda:0"):
            wp.launch(kernel, dim=1, inputs=[])
            wp.synchronize_device()

    def test_kernel_with_closure(self):
        def make_kernel(value: Index_3d):
            @wp.kernel
            def kernel():
                wp.neon_print(value)

            return kernel

        with wp.ScopedDevice("cuda:0"):
            wp.launch(make_kernel(Index_3d(-1, -2, -3)), dim=1, inputs=[])
            wp.launch(make_kernel(Index_3d(17, 42, 99)), dim=1, inputs=[])
            wp.synchronize_device()

    def test_closure_captures_neon_types(self):
        def make_kernel(idx, data_view, span, partition):
            @wp.kernel
            def kernel():
                wp.neon_print(idx)
                wp.NeonDataView_print(data_view)
                wp.NeonDenseSpan_print(span)
                wp.neon_print(partition)

            return kernel

        with wp.ScopedDevice("cuda:0"):
            backend = neon.Backend(
                runtime=neon.Backend.Runtime.stream,
                dev_idx_list=[0],
            )
            grid = neon.dense.dGrid(backend)
            field = grid.new_field(cardinality=1, dtype=wp.int32)
            partition = field.get_partition(
                neon.Execution.device(),
                0,
                neon.DataView.standard(),
            )

            span = dSpan()
            span.dataView = DataView(DataView.Values.internal)
            span.z_ghost_radius = 17
            span.z_boundary_radius = 42
            span.max_z_in_domain = 99
            span.span_dim = Index_3d(2, 4, 6)

            kernel = make_kernel(
                Index_3d(3, 2, 1),
                DataView(DataView.Values.boundary),
                span,
                partition,
            )
            wp.launch(kernel, dim=1)
            wp.synchronize_device()


if __name__ == "__main__":
    unittest.main()
