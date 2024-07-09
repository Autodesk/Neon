import copy
import ctypes
from enum import Enum

import py_neon
from .dField import dField
from py_neon.execution import Execution
from py_neon import Py_neon
from py_neon.dataview import DataView
from .dSpan import dSpan
from ..backend import Backend
from py_neon.index_3d import Index_3d
import numpy as np

import sys

class dGrid(object):

    def __init__(self, backend: Backend = None, dim: Index_3d = Index_3d(10,10,10), sparsity_pattern: np.ndarray = None): # @TODOMATT implement psarsity pattern
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.backend = backend
        self.dim = dim
        self.sparsity_pattern = sparsity_pattern
        if backend is None:
            # rise exception
            raise Exception('dGrid: backend pamrameter is missing')
        
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        
        self._help_load_api()
        self._help_grid_new()

    def __del__(self):
        if self.handle != 0:
            self._help_grid_delete()

    def _help_load_api(self):

        # grid_new
        # self.py_neon.lib.dGrid_new.argtypes = [self.py_neon.handle_type,
        #                                        self.py_neon.handle_type,
        #                                        py_neon.Index_3d]
        self.py_neon.lib.dGrid_new.argtypes = [self.py_neon.handle_type,
                                               self.py_neon.handle_type,
                                               ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.dGrid_new.restype = ctypes.c_int

        # grid_delete
        self.py_neon.lib.dGrid_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dGrid_delete.restype = ctypes.c_int

        self.py_neon.lib.dGrid_get_dimensions.argtypes = [self.py_neon.handle_type,
                                                          ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.dGrid_get_dimensions.restype = ctypes.c_int

        self.py_neon.lib.dGrid_get_span.argtypes = [self.py_neon.handle_type,
                                                    ctypes.POINTER(dSpan),  # the span object
                                                    py_neon.Execution,  # the execution type
                                                    ctypes.c_int,  # the device id
                                                    py_neon.DataView,  # the data view
                                                    ]
        self.py_neon.lib.dGrid_get_span.restype = ctypes.c_int

        self.py_neon.lib.dGrid_span_size.argtypes = [ctypes.POINTER(dSpan)]
        self.py_neon.lib.dGrid_span_size.restype = ctypes.c_int


        self.py_neon.lib.dGrid_get_properties.argtypes = [self.py_neon.handle_type,
                                                          ctypes.POINTER(py_neon.Index_3d),
                                                          ctypes.POINTER(DataView)]
        self.py_neon.lib.dGrid_get_properties.restype = ctypes.c_int

        self.py_neon.lib.dGrid_is_inside_domain.argtypes = [self.py_neon.handle_type,
                                                            ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.dGrid_is_inside_domain.restype = ctypes.c_bool


    def _help_grid_new(self):
        if self.backend.handle.value == 0:  # Check backend handle validity
            raise Exception('DGrid: Invalid backend handle')

        if self.handle.value != 0:  # Ensure the grid handle is uninitialized
            raise Exception('DGrid: Grid handle already initialized')
        
        print(f"Initializing grid with handle {self.handle.value} and backend handle {self.backend.handle.value}")
        sys.stdout.flush()  # Ensure the print statement is flushed to the console
        # idx3d = Index_3d(10,10,10)
        # res = self.py_neon.lib.dGrid_new(ctypes.byref(self.handle), ctypes.byref(self.backend.handle), idx3d)
        # res = self.py_neon.lib.dGrid_new(ctypes.byref(self.handle), ctypes.byref(self.backend.handle), self.dim.x, self.dim.y, self.dim.z)
        res = self.py_neon.lib.dGrid_new(ctypes.byref(self.handle), ctypes.byref(self.backend.handle), self.dim)
        if res != 0:
            raise Exception('DGrid: Failed to initialize grid')
        print(f"Grid initialized with handle {self.handle.value}")

    def _help_grid_delete(self):
        if self.py_neon.lib.dGrid_delete(ctypes.byref(self.handle)) != 0:
            raise Exception('Failed to delete grid')
        
    def get_python_dimensions(self):
        return self.dim
    
    def get_cpp_dimensions(self):
        cpp_dim = Index_3d(0,0,0)
        res = self.py_neon.lib.dGrid_get_dimensions(ctypes.byref(self.handle), cpp_dim)
        if res != 0:
            raise Exception('DGrid: Failed to obtain grid dimension')
        
        return cpp_dim

    def new_field(self, cardinality: ctypes.c_int) -> dField:
        field = dField(self.handle, cardinality)
        return field

    def get_span(self,
                 execution: Execution,
                 c: ctypes.c_int,
                 data_view: DataView) -> dSpan:
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        span = dSpan()
        res = self.py_neon.lib.dGrid_get_span(ctypes.byref(self.handle), span, execution, c, data_view)
        if res != 0:
            raise Exception('Failed to get span')

        cpp_size = self.py_neon.lib.dGrid_span_size(span)
        ctypes_size = ctypes.sizeof(span)

        if cpp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span
    
    def getProperties(self, idx: Index_3d):
        dv = DataView(0)
        return DataView(self.py_neon.lib.dGrid_get_properties(ctypes.byref(self.handle), idx, dv))
    
    def isInsideDomain(self, idx: Index_3d):
        return self.py_neon.lib.dGrid_is_inside_domain(ctypes.byref(self.handle), idx)
