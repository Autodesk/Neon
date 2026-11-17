import copy
import ctypes
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as _dist_version
import os
from typing import Optional
import warp as wp

try:
    __version__ = _dist_version("neon_gpu")
except PackageNotFoundError:
    # Running from a source tree without an installed neon_gpu distribution.
    __version__ = "0.0.0+local"

# from .py_ne import neon
from .gate import Gate, _find_neon_library
from .dataView import DataView
from .memoryType import MemoryType
from .execution import Execution
from .index_3d import Index_3d
from .ngh_idx import Ngh_idx

from .tool.__init__ import *
from .dense.__init__ import *
from .block.__init__ import *
from .multires.__init__ import *

from .loader import Loader
from .container import Container
from .container import container
from .container import kernel
from .timer import Timer
from .skeletonConfig import SkeletonConfig
from .skeleton import Skeleton

from .tool import report

# Import logging module
from . import logging as _logging
from .logging import logger

# Lazy-loaded library reference for logging control
_neon_lib = None


def _get_neon_lib():
    """Get the loaded Neon library (lazy initialization)."""
    global _neon_lib
    if _neon_lib is None:
        lib_path = _find_neon_library()
        _neon_lib = ctypes.CDLL(lib_path)
    return _neon_lib


def set_info_logging(enabled: bool) -> None:
    """
    Enable or disable Neon C++ INFO level logging at runtime.
    
    This controls the C++ NEON_INFO macro output.
    For Python logging, use set_python_logging() or set_log_level().
    
    Args:
        enabled: If True, C++ INFO messages will be logged. If False, they will be suppressed.
    
    Example:
        >>> import neon
        >>> neon.set_info_logging(False)  # Disable C++ INFO logging
        >>> # ... run some Neon code quietly ...
        >>> neon.set_info_logging(True)   # Re-enable C++ INFO logging
    """
    lib = _get_neon_lib()
    lib.neon_set_info_enabled(ctypes.c_int(1 if enabled else 0))


def is_info_logging_enabled() -> bool:
    """
    Check if Neon C++ INFO level logging is currently enabled.
    
    Returns:
        True if C++ INFO logging is enabled, False otherwise.
    
    Example:
        >>> import neon
        >>> neon.is_info_logging_enabled()
        True
    """
    lib = _get_neon_lib()
    lib.neon_is_info_enabled.restype = ctypes.c_int
    return lib.neon_is_info_enabled() != 0


def set_log_level(level: str) -> None:
    """
    Set the Python logging level for Neon.
    
    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    
    Example:
        >>> import neon
        >>> neon.set_log_level("DEBUG")   # Show all messages including debug
        >>> neon.set_log_level("WARNING") # Only show warnings and above
    """
    _logging.set_level(level)


def set_python_logging(enabled: bool) -> None:
    """
    Enable or disable Neon Python logging.
    
    Args:
        enabled: If True, Python log messages are shown. If False, they are suppressed.
    
    Example:
        >>> import neon
        >>> neon.set_python_logging(False)  # Disable Python logging
    """
    _logging.set_enabled(enabled)


def set_quiet(quiet: bool = True) -> None:
    """
    Convenience function to enable/disable all Neon logging (both C++ and Python).
    
    Args:
        quiet: If True, suppress all logging. If False, enable all logging.
    
    Example:
        >>> import neon
        >>> neon.set_quiet(True)   # Suppress all output
        >>> neon.set_quiet(False)  # Re-enable all output
    """
    set_info_logging(not quiet)
    set_python_logging(not quiet)


_initialized = False


def is_initialized() -> bool:
    """Return True once :func:`init` has completed in this process."""
    return _initialized


def init(clear_kernel_cache: Optional[bool] = None) -> None:
    """
    Register Neon's Warp build settings and builtins. Idempotent.

    The first call configures Warp (C++17, Neon include directory, the
    ``NEON_WARP_COMPILATION`` macro) and registers the Neon builtins. Later
    calls are no-ops, so libraries and tests can call ``neon.init()`` freely.

    Args:
        clear_kernel_cache: Clear Warp's on-disk kernel cache. ``None`` (the
            default) clears it on the first call only, unless the
            ``NEON_KEEP_KERNEL_CACHE`` environment variable is set. ``True``
            clears it even when Neon is already initialized.
    """
    global _initialized

    if _initialized:
        if clear_kernel_cache:
            wp.clear_kernel_cache()
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info(f"Initializing Neon from {script_dir}")

    wp.build.set_cpp_standard("c++17")
    wp.build.add_include_directory(script_dir)
    wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

    if clear_kernel_cache is None:
        clear_kernel_cache = not os.environ.get("NEON_KEEP_KERNEL_CACHE")
    if clear_kernel_cache:
        wp.clear_kernel_cache()

    from .warp_builtins import register_neon_warp_type
    register_neon_warp_type()

    _initialized = True