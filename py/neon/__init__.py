import copy
import ctypes
from enum import Enum
import os
import warp as wp

__version__ = "0.5.2a1"

# from .py_ne import neon
from .gate import Gate
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