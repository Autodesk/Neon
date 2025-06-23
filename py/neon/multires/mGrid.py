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
                 backend: neon.Backend,
                 dim,
                 sparsity_pattern_list: List[np.ndarray],
                 sparsity_pattern_origins: List[neon.Index_3d],
                 stencil: List[List[int]]):

        if backend is None:
            # raise exception
            raise Exception('dGrid: backend parameter is missing')

        # for sparsity_pattern in sparsity_pattern_list:
        #     if (sparsity_pattern.shape[0] != dim.x or
        #             sparsity_pattern.shape[1] != dim.y or
        #             sparsity_pattern.shape[2] != dim.z):
        #         raise Exception('mGrid: sparsity_pattern\'s shape does not match the dim')

        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.backend = backend
        self.dim = dim
        self.sparsity_pattern_list = sparsity_pattern_list
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.stencil = stencil
        self.num_levels = len(sparsity_pattern_list)

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
                                 ctypes.c_int,
                                 ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                 ctypes.POINTER(Index_3d),
                                 ctypes.POINTER(Index_3d),
                                 ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int)]
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
                                      ctypes.c_int,  # the grid level
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
        def prepare_int32_arrays_and_sizes(np_arrays):
            """
            Given a list of 3D NumPy arrays of type np.int32, returns:
              - c_arrays: a ctypes array of pointers to the data of each array.
              - dims0, dims1, dims2: ctypes arrays (of type c_int) containing the sizes in each dimension.

            Each array is made contiguous in memory to ensure proper layout.
            """
            # Ensure each array is contiguous in memory.
            contiguous_arrays = [np.ascontiguousarray(arr) for arr in np_arrays]

            num_arrays = len(contiguous_arrays)

            # Define a ctypes pointer type for int32_t.
            # ctypes.c_int32 represents a 32-bit integer.
            Int32P = ctypes.POINTER(ctypes.c_int32)

            # Create a ctypes array type that can hold 'num_arrays' pointers.
            ArrayOfPointers = Int32P * num_arrays
            # Build the ctypes array of pointers by converting each NumPy array's data pointer.
            c_arrays = ArrayOfPointers(*(arr.ctypes.data_as(Int32P) for arr in contiguous_arrays))

            ArraysOfIndex3d = Index_3d * num_arrays
            # Create ctypes arrays for each dimension.
            dims_array = ArraysOfIndex3d()
            for idx, arr in enumerate(contiguous_arrays):
                dims_array[idx] = Index_3d(arr.shape[0], arr.shape[1], arr.shape[2])

            origin_array_type = Index_3d * len(self.sparsity_pattern_origins)
            origin_array = origin_array_type()
            for idx, origin in enumerate(self.sparsity_pattern_origins):
                origin_array[idx] = origin

            return num_arrays, c_arrays, dims_array, origin_array

        if self.backend.backend_handle.value == ctypes.c_void_p(0):  # Check backend handle validity
            raise Exception('mGrid: Invalid backend handle')

        if self.handle.value != None:  # Ensure the grid handle is uninitialized
            raise Exception('mGrid: Grid handle already initialized')

        num_arrays, c_arrays, dims, origins = prepare_int32_arrays_and_sizes(self.sparsity_pattern_list)
        self.depth = num_arrays

        stencil_type = ctypes.c_int * (3 * len(self.stencil))
        stencil_array = stencil_type()
        for s_idx, s in enumerate(self.stencil):
            a_idx = s_idx * 3
            stencil_array[a_idx] = s[0]
            stencil_array[a_idx + 1] = s[1]
            stencil_array[a_idx + 2] = s[2]

        res = self.api_new(ctypes.pointer(self.handle),
                           self.backend.backend_handle,
                           self.dim,
                           self.depth,
                           c_arrays, dims, origins,
                           len(self.stencil),
                           stencil_array)
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
        cpp_dim = Index_3d(0, 0, 0)
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
            raise Exception(
                'can\'t access negative indices in mGrid')  # @TODOMATT make sure that this is a valid requirement
        return self.neon.lib.mGrid_is_inside_domain(self.handle, grid_level, idx)

    def get_backend(self):
        return self.backend

    def get_handle(self):
        return self.handle

    def get_name(self):
        return "mGrid"

    def get_num_levels(self):
        return self.num_levels

    def get_dimensions(self):
        return self.dim
