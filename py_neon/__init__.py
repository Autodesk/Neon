import copy
import ctypes
from enum import Enum

from .py_ne import Py_neon
from .dataview import DataView
from .execution import Execution
from .index_3d import Index_3d

from .dense.__init__ import *

#
#
# class PyNeon(object):
#     def __init__(self):
#         self.handle_type = ctypes.POINTER(ctypes.c_uint64)
#         self.lib = ctypes.CDLL(
#             '/home/max/repos/neon/warp/neon_warp_testing/neon_py_bindings/cmake-build-debug/libNeonPy/liblibNeonPy.so')
#         # # grid_new
#         # self.lib.grid_new.argtypes = [self.handle_type]
#         # # self.lib.grid_new.re = [ctypes.c_int]
#         # # grid_delete
#         # self.lib.grid_delete.argtypes = [self.handle_type]
#         # # self.lib.grid_delete.restype = [ctypes.c_int]
#         # # new_field
#         # self.lib.field_new.argtypes = [self.handle_type, self.handle_type]
#         # # delete_field
#         # self.lib.field_delete.argtypes = [self.handle_type]
#
#     def field_new(self, handle_field: ctypes.c_uint64, handle_grid: ctypes.c_uint64):
#         res = self.lib.field_new(handle_field, handle_grid)
#         if res != 0:
#             raise Exception('Failed to initialize field')
#
#     def field_delete(self, handle_field: ctypes.c_uint64):
#         res = self.lib.grid_delete(handle_field)
#         if res != 0:
#             raise Exception('Failed to initialize grid')
#
