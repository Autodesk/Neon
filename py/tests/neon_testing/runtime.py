"""Process-wide Warp/Neon setup, GPU detection, skip decorators and artifacts."""

import os
import unittest

import warp as wp

import neon

CONTAINER_RUNTIME = neon.Container.ContainerRuntime.neon

_initialized = False


def init_once() -> None:
    """Initialize Warp and Neon once per process.

    Warp runs in debug mode by default; set ``NEON_TEST_WARP_MODE=release`` for
    faster kernel compilation. ``NEON_TEST_VERBOSE=1`` enables Warp verbosity.
    """
    global _initialized
    if _initialized:
        return
    mode = os.environ.get("NEON_TEST_WARP_MODE", "debug")
    wp.config.mode = mode
    if mode == "debug":
        wp.config.llvm_cuda = False
    if os.environ.get("NEON_TEST_VERBOSE"):
        wp.config.verbose = True
        wp.config.verbose_warnings = True
    wp.init()
    neon.init()
    _initialized = True


_gpu_count = None
_gpu_count_error = None


def gpu_count() -> int:
    """Number of CUDA devices Warp can see (cached). 0 if CUDA is unavailable."""
    global _gpu_count, _gpu_count_error
    if _gpu_count is None:
        try:
            _gpu_count = wp.get_cuda_device_count()
        except RuntimeError as exc:
            _gpu_count, _gpu_count_error = 0, exc
    return _gpu_count


def require_gpus(n: int = 1):
    """Skip a test or class unless at least ``n`` CUDA devices are available."""
    count = gpu_count()
    reason = f"needs {n} CUDA GPU(s), found {count}"
    if _gpu_count_error is not None:
        reason += f" ({_gpu_count_error})"
    return unittest.skipUnless(count >= n, reason)


require_gpu = require_gpus(1)


def run_container(container, stream_idx: int = 0) -> None:
    """Run a container on the Neon runtime and wait for it."""
    container.run(
        stream_idx=stream_idx,
        data_view=neon.DataView.standard(),
        container_runtime=CONTAINER_RUNTIME,
    )
    wp.synchronize()


def artifact_dir():
    """Directory for optional test outputs (``NEON_TEST_ARTIFACTS``), or None."""
    path = os.environ.get("NEON_TEST_ARTIFACTS")
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def export_vti_if_requested(field, name: str, **kwargs) -> None:
    """Export ``field`` to ``$NEON_TEST_ARTIFACTS/<name>`` when ``NEON_EXPORT_VTI`` is set."""
    if not os.environ.get("NEON_EXPORT_VTI"):
        return
    out = artifact_dir() or os.getcwd()
    field.export_vti(os.path.join(out, name), **kwargs)


def non_primary_cuda_devices():
    """Warp devices bound to a non-primary CUDA context.

    Uses Warp internals; raises ``unittest.SkipTest`` if they are not available.
    """
    try:
        from warp._src import context as wp_ctx

        devices = wp_ctx.runtime.cuda_devices
        return [d.alias for d in devices if d.is_cuda and not d.is_primary]
    except (ImportError, AttributeError) as exc:
        raise unittest.SkipTest(f"Warp device internals unavailable: {exc}")


# --- optional bindings -------------------------------------------------------

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


def _has_symbols(*symbols) -> bool:
    lib = neon.Gate().lib
    for symbol in symbols:
        if not hasattr(lib, symbol):
            return False
    return True


def mres_fp16_available() -> bool:
    if wp.float16 not in neon.Gate().warp_type_to_string:
        return False
    if not hasattr(neon.multires.mPartition, "mPartition_float16"):
        return False
    return _has_symbols(*MRES_FP16_REQUIRED_SYMBOLS)


def mres_fp16_container_available() -> bool:
    return mres_fp16_available() and _has_symbols(MRES_FP16_CONTAINER_SYMBOL)
