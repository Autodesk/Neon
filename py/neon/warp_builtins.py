import copy
import ctypes
from enum import Enum
import os
import warp as wp

# from .py_ne import neon
from .gate import Gate
from .dataview import DataView
from .execution import Execution
from .index_3d import Index_3d
from .ngh_idx import Ngh_idx

from .tool.__init__ import *
from .dense.__init__ import *
from .block.__init__ import *

from .loader import Loader
from .container import Container
from .timer import Timer
from .skeleton import Skeleton


def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive


def _register_base_builtins():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/Index_3d.h")
    _add_header(f"{include_path}/dDataView.h")
    _add_header(f"{include_path}/ngh_idx.h")
    Index_3d.warp_register_builtins()
    DataView.warp_register_builtins()
    Ngh_idx.warp_register_builtins()


def _register_dense_builtins():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/dense/dSpan.h")
    _add_header(f"{include_path}/dense/dPartition.h")
    _add_header(f"{include_path}/dense/dIndex.h")
    from .dense import dIndex, dSpan, dPartition
    dIndex.register_builtins()
    dSpan.register_builtins()
    dPartition.register_builtins()

def _register_block_builtins():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/block/bSpan.h")
    _add_header(f"{include_path}/block/bPartition.h")
    _add_header(f"{include_path}/block/bIndex.h")
    from .block import bIndex, bSpan, bPartition
    bIndex.register_builtins()
    bSpan.register_builtins()
    bPartition.register_builtins()

def register_neon_warp_type():
    _register_base_builtins()
    _register_dense_builtins()
    _register_block_builtins()
