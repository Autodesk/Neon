import ctypes

import numpy as np

import py_neon
from py_neon import Py_neon
from py_neon.dataview import DataView
from py_neon.execution import Execution
from py_neon.index_3d import Index_3d
from .dField import dField
from .dSpan import dSpan
from ..backend import Backend


class dGrid(object):

    def __init__(self, backend: Backend = None, dim: Index_3d = Index_3d(10, 10, 10),
                 sparsity_pattern: np.ndarray = None):
        if sparsity_pattern is None:
            sparsity_pattern = np.ones((dim.x, dim.y, dim.z))
        if backend is None:
            # raise exception
            raise Exception('dGrid: backend pamrameter is missing')
        if sparsity_pattern.shape[0] != dim.x or sparsity_pattern.shape[1] != dim.y or sparsity_pattern.shape[
            2] != dim.z:
            raise Exception('dGrid: sparsity_pattern\'s shape does not match the dim')

        self.grid_handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.backend = backend
        self.dim = dim
        self.sparsity_pattern = sparsity_pattern

        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.grid_handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))

        self._help_load_api()
        self._help_grid_new()

    def __del__(self):
        if self.grid_handle != 0:
            self._help_grid_delete()

    def _help_load_api(self):

        self.py_neon.lib.dGrid_new.argtypes = [self.py_neon.handle_type,
                                               self.py_neon.handle_type,
                                               ctypes.POINTER(py_neon.Index_3d),
                                               ctypes.POINTER(ctypes.c_int)]
        self.py_neon.lib.dGrid_new.restype = ctypes.c_int
        # ---
        self.py_neon.lib.dGrid_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dGrid_delete.restype = ctypes.c_int
        # ---
        self.py_neon.lib.dGrid_get_dimensions.argtypes = [self.py_neon.handle_type,
                                                          ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.dGrid_get_dimensions.restype = ctypes.c_int
        # ---
        self.py_neon.lib.dGrid_get_span.argtypes = [self.py_neon.handle_type,
                                                    ctypes.POINTER(dSpan),  # the span object
                                                    py_neon.Execution,  # the execution type
                                                    ctypes.c_int,  # the device id
                                                    py_neon.DataView,  # the data view
                                                    ]
        self.py_neon.lib.dGrid_get_span.restype = ctypes.c_int
        # ---
        self.py_neon.lib.dGrid_span_size.argtypes = [ctypes.POINTER(dSpan)]
        self.py_neon.lib.dGrid_span_size.restype = ctypes.c_int
        # ---
        self.py_neon.lib.dGrid_get_properties.argtypes = [self.py_neon.handle_type,
                                                          ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.dGrid_get_properties.restype = ctypes.c_int
        # ---
        self.py_neon.lib.dGrid_is_inside_domain.argtypes = [self.py_neon.handle_type,
                                                            ctypes.POINTER(py_neon.Index_3d)]
        self.py_neon.lib.dGrid_is_inside_domain.restype = ctypes.c_bool

    def _help_grid_new(self):

        if self.grid_handle.value != 0:  # Ensure the grid handle is uninitialized
            raise Exception('dGrid: Grid handle already initialized')

        sparsity_pattern_array = self.sparsity_pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        res = self.py_neon.lib.dGrid_new(ctypes.byref(self.grid_handle),
                                         ctypes.byref(self.backend.backend_handle),
                                         self.dim,
                                         sparsity_pattern_array)
        if res != 0:
            raise Exception('dGrid: Failed to initialize grid')
        print(f"dGrid initialized with handle {self.grid_handle.value}")

    def _help_grid_delete(self):
        if self.py_neon.lib.dGrid_delete(ctypes.byref(self.grid_handle)) != 0:
            raise Exception('Failed to delete grid')

    def get_python_dimensions(self):
        return self.dim

    def get_cpp_dimensions(self):
        cpp_dim = Index_3d(0, 0, 0)
        res = self.py_neon.lib.dGrid_get_dimensions(ctypes.byref(self.grid_handle), cpp_dim)
        if res != 0:
            raise Exception('DGrid: Failed to obtain grid dimension')

        return cpp_dim

    def new_field(self, cardinality: int) -> dField:
        cardinality = ctypes.c_int(cardinality)
        field = dField(py_neon=self.py_neon,
                       grid_handle=self.grid_handle,
                       cardinality=cardinality,
                       py_grid=self)
        return field

    def get_span(self,
                 execution: Execution,
                 dev_idx: int,
                 data_view: DataView) -> dSpan:
        if self.grid_handle == 0:
            raise Exception('DGrid: Invalid handle')

        span = dSpan()
        dev_idx_ctypes = ctypes.c_int(dev_idx)
        res = self.py_neon.lib.dGrid_get_span(ctypes.byref(self.grid_handle),
                                              span,
                                              execution,
                                              dev_idx_ctypes,
                                              data_view)
        if res != 0:
            raise Exception('Failed to get span')

        cpp_size = self.py_neon.lib.dGrid_span_size(span)
        ctypes_size = ctypes.sizeof(span)

        if cpp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span

    def get_properties(self, idx: Index_3d):
        return DataView(self.py_neon.lib.dGrid_get_properties(ctypes.byref(self.grid_handle), idx))

    def is_inside_domain(self, idx: Index_3d):
        return self.py_neon.lib.dGrid_is_inside_domain(ctypes.byref(self.grid_handle), idx)

    def get_backend(self):
        return self.backend

    def get_handle(self):
        return self.grid_handle