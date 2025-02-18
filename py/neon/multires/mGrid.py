import ctypes

import neon
from .mField import mField
from neon.execution import Execution
from neon.dataview import DataView
from ..block.bSpan import bSpan
from neon.index_3d import Index_3d
import numpy as np
from typing import List

class mGrid(object):
    def __init__(self,
                 backend : neon.Backend,
                 dim ,
                 depth : int,
                 sparsity_pattern_list: List[np.ndarray]):
        # the lenght of the spartiry_pattern_list should be equal to the depth
        if len(sparsity_pattern_list) != depth:
            raise Exception('mGrid: sparsity_pattern_list\'s length does not match the depth')

        if backend is None:
            # raise exception
            raise Exception('dGrid: backend parameter is missing')

        for sparsity_pattern in sparsity_pattern_list:
            if (sparsity_pattern.shape[0] != dim.x or
                    sparsity_pattern.shape[1] != dim.y or
                    sparsity_pattern.shape[2] != dim.z):
                raise Exception('mGrid: sparsity_pattern\'s shape does not match the dim')


        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.backend = backend
        self.dim = dim
        self.sparsity_pattern_list = sparsity_pattern_list
        self.depth = depth


        self._help_load_api()
        self._help_grid_new()

    def __del__(self):
        if self.handle == 0:
            return
        self._help_grid_delete()

    def _help_load_api(self):
        self.neon_gate: neon.Gate = neon.Gate()
        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.handle_type = ctypes.c_void_p

        lib = self.neon_gate.lib
        # grid_new
        self.api_new = lib.mGrid_new
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),
                                 self.handle_type,
                                 ctypes.POINTER(neon.Index_3d),
                                 ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                 ctypes.c_int,
                                 ctypes.c_int]
        self.api_new.restype = ctypes.c_int

        # grid_delete
        self.api_delete = lib.mGrid_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.api_delete.restype = ctypes.c_int

        self.api_get_dimensions = lib.mGrid_get_dimensions
        self.api_get_dimensions.argtypes = [self.handle_type,
                                            ctypes.POINTER(neon.Index_3d)]
        self.api_get_dimensions.restype = ctypes.c_int

        self.api_get_span = lib.mGrid_get_span
        self.api_get_span.argtypes = [self.handle_type,
                                      ctypes.c_int, # the grid level
                                      ctypes.POINTER(bSpan),  # the span object
                                      neon.Execution,  # the execution type
                                      ctypes.c_int,  # the device id
                                      neon.DataView,  # the data view
                                      ]
        self.api_get_span.restype = ctypes.c_int

        self.api_is_inside_domain = lib.mGrid_is_inside_domain
        self.api_is_inside_domain.argtypes = [self.handle_type,
                                              ctypes.c_int,
                                              ctypes.POINTER(neon.Index_3d)]
        self.api_is_inside_domain.restype = ctypes.c_bool


    def _help_grid_new(self):
        def get_numpy_array_pointers(arrays):
            """
            Given a list of NumPy arrays, returns a ctypes pointer to a contiguous
            block of pointers (of type `double**`) to the data buffers of these arrays.

            Parameters:
                arrays (list of np.ndarray): List of NumPy arrays (assumed to be float64).

            Returns:
                (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), int):
                    A tuple containing:
                      - A pointer to the first element of a contiguous block containing pointers
                        to each array's data buffer.
                      - The count of arrays (i.e., number of pointers).

            Note:
                The original NumPy arrays must remain alive as long as the returned pointer is used.
            """
            pointer_values = np.array([arr.ctypes.data for arr in arrays], dtype=np.intp)
            c_pointer = pointer_values.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))
            return c_pointer, len(arrays)

        if self.backend.backend_handle.value == ctypes.c_void_p(0):  # Check backend handle validity
            raise Exception('mGrid: Invalid backend handle')

        if self.handle.value != None:  # Ensure the grid handle is uninitialized
            raise Exception('mGrid: Grid handle already initialized')

        sparsity_pattern_array, sparsity_pattern_array_size = get_numpy_array_pointers(self.sparsity_pattern_list)

        res = self.api_new(ctypes.pointer(self.handle),
                           self.backend.backend_handle,
                           self.dim,
                           sparsity_pattern_array,
                           sparsity_pattern_array_size,
                           self.depth)
        if res != 0:
            raise Exception('mGrid: Failed to initialize grid')
        print(f"mGrid initialized with handle {self.handle.value}")

    def _help_grid_delete(self):
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete grid')

    def get_python_dimensions(self):
        return self.dim

    def get_cpp_dimensions(self):
        cpp_dim = Index_3d(0,0,0)
        res = self.neon.lib.mGrid_get_dimensions(self.handle, cpp_dim)
        if res != 0:
            raise Exception('mGrid: Failed to obtain grid dimension')

        return cpp_dim

    def new_field(self,
                  cardinality: ctypes.c_int,
                  dtype) -> mField:
        field = mField(neon_gate=self.neon_gate,
                       grid_handle=self.handle,
                       cardinality=cardinality,
                       py_grid=self,
                       dtype=dtype
                       )
        return field

    def get_span(self,
                 grid_level: ctypes.c_int,
                 execution: Execution,
                 dev_idx: ctypes.c_int,
                 data_view: DataView) -> bSpan:
        if self.handle == 0:
            raise Exception('mGrid: Invalid handle')

        span = bSpan()
        res = self.api_get_span(self.handle, grid_level, span, execution, dev_idx, data_view)
        if res != 0:
            raise Exception('Failed to get span')

        # cpp_size = self.neon.lib.mGrid_span_size(span)
        # ctypes_size = ctypes.sizeof(span)
        #
        # if cpp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span

    def getProperties(self, grid_level: ctypes.c_int, idx: Index_3d):
        return DataView(self.neon.lib.mGrid_get_properties(self.handle, grid_level, idx))

    def isInsideDomain(self, grid_level: ctypes.c_int, idx: Index_3d):
        if idx.x < 0 or idx.y < 0 or idx.z < 0:
            raise Exception('can\'t access negative indices in mGrid') # @TODOMATT make sure that this is a valid requirement
        return self.neon.lib.mGrid_is_inside_domain(self.handle, grid_level, idx)

    def get_backend(self):
        return self.backend

    def get_handle(self):
        return self.handle

    def get_name(self):
        return "mGrid"