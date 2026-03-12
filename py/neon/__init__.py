import copy
import ctypes
from enum import Enum
import os
import warp as wp

__version__ = "0.5.2a1"

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
    Enable or disable Neon INFO level logging at runtime.
    
    Args:
        enabled: If True, INFO messages will be logged. If False, they will be suppressed.
    
    Example:
        >>> import neon
        >>> neon.set_info_logging(False)  # Disable INFO logging
        >>> # ... run some Neon code quietly ...
        >>> neon.set_info_logging(True)   # Re-enable INFO logging
    """
    lib = _get_neon_lib()
    lib.neon_set_info_enabled(ctypes.c_int(1 if enabled else 0))


def is_info_logging_enabled() -> bool:
    """
    Check if Neon INFO level logging is currently enabled.
    
    Returns:
        True if INFO logging is enabled, False otherwise.
    
    Example:
        >>> import neon
        >>> neon.is_info_logging_enabled()
        True
    """
    lib = _get_neon_lib()
    lib.neon_is_info_enabled.restype = ctypes.c_int
    return lib.neon_is_info_enabled() != 0


def init():
    # Get the path of the current script
    script_path = __file__

    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    print(f"Directory containing the script: {script_dir}")

    wp.build.set_cpp_standard("c++17")
    wp.build.add_include_directory(script_dir)
    wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

    # It's a good idea to always clear the kernel cache when developing new native or codegen features
    wp.build.clear_kernel_cache()

    from .warp_builtins import register_neon_warp_type
    register_neon_warp_type()