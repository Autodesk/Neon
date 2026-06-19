"""Shared helpers for py/tests."""

import os
import unittest

import numpy as np
import warp as wp
import neon

CONTAINER_RUNTIME = neon.Container.ContainerRuntime.neon
DATA_VIEW = neon.DataView.standard()

REF_FACTOR = 2
GRID_DIM = 16
SHELL_DEPTH = 4
LEVEL_ONE_PEEL = 2

FP16_TOLERANCE = 1.0e-2


def init_warp_neon(*, debug=True, verbose=False):
    if debug:
        wp.config.mode = "debug"
        wp.config.llvm_cuda = False
    if verbose:
        wp.config.verbose = True
        wp.verbose_warnings = True
    wp.init()
    neon.init()


_WARP_NEON_READY = False


def init_warp_neon_once(*, debug=True, verbose=False):
    """Initialize Warp/Neon once per process (neon.init clears the kernel cache)."""
    global _WARP_NEON_READY
    if _WARP_NEON_READY:
        return
    init_warp_neon(debug=debug, verbose=verbose)
    _WARP_NEON_READY = True


def run_container(container, stream_idx=0):
    container.run(
        stream_idx=stream_idx,
        data_view=DATA_VIEW,
        container_runtime=CONTAINER_RUNTIME,
    )
    wp.synchronize()


def coord_sum(idx):
    return idx.x + idx.y + idx.z


def gpu_count():
    try:
        return wp.get_cuda_device_count()
    except Exception:
        return 0


def gpu_available():
    return gpu_count() > 0


def wpne_available():
    try:
        import wpne  # noqa: F401
        return True
    except ImportError:
        return False


MRES_FP16_REQUIRED_SYMBOLS = (
    "mGrid_mField_new_float16",
    "mGrid_mField_delete_float16",
    "mGrid_mField_get_partition_float16",
    "mGrid_mField_read_float16",
    "mGrid_mField_write_float16",
    "mGrid_mField_update_host_data_float16",
    "mGrid_mField_update_device_data_float16",
    "mGrid_mField_fill_float16",
    "mGrid_mField_copy_float16",
)

MRES_FP16_CONTAINER_SYMBOL = "warp_container_mres_add_parse_token_mGrid_float16_0"


def mres_fp16_available() -> bool:
    try:
        gate = neon.Gate()
        gate.get_type_mapping(wp.float16)
        getattr(neon.multires.mPartition, "mPartition_float16")
        for symbol in MRES_FP16_REQUIRED_SYMBOLS:
            getattr(gate.lib, symbol)
        return True
    except (AttributeError, Exception):
        return False


def mres_fp16_container_available() -> bool:
    if not mres_fp16_available():
        return False
    try:
        getattr(neon.Gate().lib, MRES_FP16_CONTAINER_SYMBOL)
        return True
    except AttributeError:
        return False


def setup_wpne_build(script_dir):
    import wpne

    wp.build.set_cpp_standard("c++17")
    wp.build.add_include_directory(script_dir)
    wp.build.add_preprocessor_macro_definition("NEON_WARP_COMPILATION")
    wp.build.clear_kernel_cache()
    wpne.init()


def make_shell_mask(dim, shell_depth):
    mask = np.zeros((dim, dim, dim), dtype=np.int32)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for depth in range(shell_depth):
                    if (i in (depth, dim - 1 - depth) or
                            j in (depth, dim - 1 - depth) or
                            k in (depth, dim - 1 - depth)):
                        mask[i, j, k] = 1
                        break
    return np.ascontiguousarray(mask, dtype=np.int32)


def make_peeled_mask(coarse_dim, peel_depth):
    mask = np.ones((coarse_dim, coarse_dim, coarse_dim), dtype=np.int32)
    for i in range(coarse_dim):
        for j in range(coarse_dim):
            for k in range(coarse_dim):
                for depth in range(peel_depth):
                    if (i in (depth, coarse_dim - 1 - depth) or
                            j in (depth, coarse_dim - 1 - depth) or
                            k in (depth, coarse_dim - 1 - depth)):
                        mask[i, j, k] = 0
                        break
    return np.ascontiguousarray(mask, dtype=np.int32)


def strip_level_zero_under_level_one(level_zero_mask, level_one_mask, block_spacing, child_span):
    coarse_dim = level_one_mask.shape[0]
    for bz in range(coarse_dim):
        for by in range(coarse_dim):
            for bx in range(coarse_dim):
                if level_one_mask[bx, by, bz] == 0:
                    continue
                ox = bx * block_spacing
                oy = by * block_spacing
                oz = bz * block_spacing
                level_zero_mask[
                    ox:ox + child_span,
                    oy:oy + child_span,
                    oz:oz + child_span,
                ] = 0
    return np.ascontiguousarray(level_zero_mask, dtype=np.int32)


def export_vti_if_requested(field, filename, **kwargs):
    if os.environ.get("NEON_EXPORT_VTI"):
        field.export_vti(filename, **kwargs)


def require_gpu(test_item):
    return unittest.skipUnless(gpu_available(), "CUDA GPU not available")(test_item)


def require_wpne(test_item):
    return unittest.skipUnless(wpne_available(), "wpne module not available")(test_item)
