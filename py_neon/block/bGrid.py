import copy
import ctypes
from enum import Enum

import py_neon
from .bField import bField
from py_neon.execution import Execution
from py_neon import Py_neon
from py_neon.dataview import DataView
from .bSpan import bSpan
from ..backend import Backend
from py_neon.index_3d import Index_3d

class bGrid(object):


    def __init__(self, backend = None, dim = None):
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.backend = backend
        self.dim = dim
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()
        self._help_grid_new()

    def __del__(self):
        if self.handle == 0:
            return
        self._help_grid_delete()

    def _help_load_api(self):

        # grid_new
        self.py_neon.lib.bGrid_new.argtypes = [self.py_neon.handle_type,
                                               self.py_neon.handle_type,
                                               ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.bGrid_new.restype = ctypes.c_int

        # grid_delete
        self.py_neon.lib.bGrid_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.bGrid_delete.restype = ctypes.c_int

        self.py_neon.lib.bGrid_get_span.argtypes = [self.py_neon.handle_type,
                                                    ctypes.POINTER(bSpan),  # the span object
                                                    py_neon.Execution,  # the execution type
                                                    ctypes.c_int,  # the device id
                                                    py_neon.DataView,  # the data view
                                                    ]
        self.py_neon.lib.bGrid_get_span.restype = ctypes.c_int

        self.py_neon.lib.bGrid_span_size.argtypes = [ctypes.POINTER(bSpan)]
        self.py_neon.lib.bGrid_span_size.restype = ctypes.c_int


        self.py_neon.lib.bGrid_get_properties.argtypes = [self.py_neon.handle_type,
                                                          ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.bGrid_get_properties.restype = ctypes.c_int

        self.py_neon.lib.bGrid_is_inside_domain.argtypes = [self.py_neon.handle_type,
                                                            ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.bGrid_is_inside_domain.restype = ctypes.c_bool


    def _help_grid_new(self):
        if self.handle == 0:
            raise Exception('bGrid: Invalid handle')

        if self.backend is None:
            self.backend = Backend()
        if self.dim is None:
            self.dim = py_neon.Index_3d(10,10,10)

        res = self.py_neon.lib.bGrid_new(ctypes.byref(self.handle), ctypes.byref(self.backend.handle), self.dim)
        if res != 0:
            raise Exception('bGrid: Failed to initialize grid')

    def _help_grid_delete(self):
        if self.py_neon.lib.bGrid_delete(ctypes.byref(self.handle)) != 0:
            raise Exception('Failed to delete grid')

    def new_field(self) -> bField:
        field = bField(self.py_neon, self.handle)
        return field

    def get_span(self,
                 execution: Execution,
                 c: ctypes.c_int,
                 data_view: DataView) -> bSpan:
        if self.handle == 0:
            raise Exception('bGrid: Invalid handle')

        span = bSpan()
        res = self.py_neon.lib.bGrid_get_span(ctypes.byref(self.handle), span, execution, c, data_view)
        if res != 0:
            raise Exception('Failed to get span')

        cpp_size = self.py_neon.lib.bGrid_span_size(span)
        ctypes_size = ctypes.sizeof(span)

        if cpp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span
    
    def getProperties(self, idx: Index_3d):
        return DataView.from_int(self.py_neon.lib.bGrid_get_properties(ctypes.byref(self.handle), idx))
    
    def isInsideDomain(self, idx: Index_3d):
        return self.py_neon.lib.bGrid_is_inside_domain(ctypes.by_ref(self.handle), idx)
