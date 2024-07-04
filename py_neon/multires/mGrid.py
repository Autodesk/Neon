import copy
import ctypes
from enum import Enum

import py_neon
from .mField import mField
from py_neon.execution import Execution
from py_neon import Py_neon
from py_neon.dataview import DataView
from .dSpan import dSpan
from .backend import Backend
from py_neon.index_3d import Index_3d

class mGrid(object):


    def __init__(self, backend = None, dim = None, depth = 0):
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.backend = backend
        self.dim = dim
        self.depth = depth
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_grid_new()

    def __del__(self):
        if self.handle == 0:
            return
        self.help_grid_delete()

    def help_load_api(self):

        # grid_new
        self.py_neon.lib.mGrid_new.argtypes = [self.py_neon.handle_type,
                                               ctypes.POINTER(Backend),
                                               py_neon.Index_3d,
                                               ctypes.c_int]
        self.py_neon.lib.mGrid_new.restype = ctypes.c_int

        # grid_delete
        self.py_neon.lib.mGrid_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.mGrid_delete.restype = ctypes.c_int

        self.py_neon.lib.mGrid_get_span.argtypes = [self.py_neon.handle_type,
                                                    ctypes.c_int, # the grid index
                                                    ctypes.POINTER(dSpan),  # the span object
                                                    py_neon.Execution,  # the execution type
                                                    ctypes.c_int,  # the device id
                                                    py_neon.DataView,  # the data view
                                                    ]
        self.py_neon.lib.mGrid_get_span.restype = ctypes.c_int

        self.py_neon.lib.mGrid_span_size.argtypes = [ctypes.POINTER(dSpan)]
        self.py_neon.lib.mGrid_span_size.restype = ctypes.c_int


        self.py_neon.lib.mGrid_get_properties.argtypes = [self.py_neon.handle_type,
                                                          ctypes.c_int, # the grid index
                                                          py_neon.Index_3d]
        self.py_neon.lib.mGrid_get_properties.restype = ctypes.c_int

        self.py_neon.lib.mGrid_is_inside_domain.argtypes = [self.py_neon.handle_type,
                                                            ctypes.c_int,
                                                            py_neon.Index_3d]
        self.py_neon.lib.mGrid_is_inside_domain.restype = ctypes.c_bool


    def help_grid_new(self):
        if self.handle == 0:
            raise Exception('mGrid: Invalid handle')

        if self.backend is None:
            self.backend = Backend()
        if self.dim is None:
            self.dim = py_neon.Index_3d(10,10,10)

        res = self.py_neon.lib.mGrid_new(self.handle, self.backend.handle, self.dim, self.depth)
        if res != 0:
            raise Exception('mGrid: Failed to initialize grid')

    def help_grid_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.mGrid_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete grid')

    def new_field(self) -> mField:
        field = mField(self.py_neon, self.handle)
        return field

    def get_span(self,
                 grid_level: ctypes.c_int,
                 execution: Execution,
                 c: ctypes.c_int,
                 data_view: DataView) -> dSpan:
        if self.handle == 0:
            raise Exception('mGrid: Invalid handle')

        span = dSpan()
        res = self.py_neon.lib.mGrid_get_span(self.handle, grid_level, span, execution, c, data_view)
        if res != 0:
            raise Exception('Failed to get span')

        cpp_size = self.py_neon.lib.mGrid_span_size(span)
        ctypes_size = ctypes.sizeof(span)

        if cpp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span
    
    def getProperties(self, grid_inex: ctypes.c_int, idx: Index_3d):
        return DataView.from_int(self.py_neon.lib.mGrid_get_properties(self.handle, grid_inex, idx))
    
    def isInsideDomain(self, grid_level: ctypes.c_int, idx: Index_3d):
        return self.py_neon.lib.mGrid_is_inside_domain(self.handle, grid_level, idx)
