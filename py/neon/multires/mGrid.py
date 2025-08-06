"""
Multi-resolution Grid Implementation for Neon Computing Framework

This module provides the mGrid class, which represents a multi-resolution
hierarchical grid data structure for parallel computing applications. The grid
supports adaptive mesh refinement patterns and enables efficient computation
across multiple resolution levels.

Author: Neon Team
"""

import ctypes

import neon
from .mField import mField
from neon.execution import Execution
from neon.dataView import DataView
from ..block.bSpan import bSpan
from neon.index_3d import Index_3d
import numpy as np
from typing import List


class mGrid(object):
    """
    Multi-resolution Grid for hierarchical parallel computing.
    
    The mGrid class represents a multi-resolution structure created by stacking
    multiple block grids at different resolution levels. Each level contains
    sparse block arrangements defined by sparsity patterns, enabling efficient
    multi-scale computation patterns commonly used in scientific computing.
    
    Key Features:
    - Multi-resolution hierarchy created by stacking block grids
    - Sparse block representation using sparsity patterns per level
    - Support for arbitrary stencil patterns across all levels
    - Cross-platform execution (CPU/GPU)
    - Integration with field data structures
    - Optimized for parallel execution across resolution levels
    
    The multi-resolution structure is defined by:
    - Base dimensions covering the computational domain
    - Per-level sparsity patterns (3D numpy arrays) defining active blocks
    - Origin points for each level's coordinate system
    - Stencil patterns defining computational neighborhoods
    
    Attributes:
        handle: C++ object handle for the grid
        backend: Neon backend configuration
        dim: Base grid dimensions (Index_3d)
        sparsity_pattern_list: List of 3D numpy arrays defining active blocks per level
        sparsity_pattern_origins: List of origin points for each level
        stencil: List of 3D offset patterns for computational stencils
        num_levels: Number of resolution levels in the stacked structure
    """
    def __init__(self,
                 backend: neon.Backend,
                 dim,
                 sparsity_pattern_list: List[np.ndarray],
                 sparsity_pattern_origins: List[neon.Index_3d],
                 stencil: List[List[int]]):
        """
        Initialize a multi-resolution grid with hierarchical structure.
        
        Args:
            backend (neon.Backend): Neon backend configuration for execution context
            dim (neon.Index_3d): Base dimensions of the computational domain
            sparsity_pattern_list (List[np.ndarray]): List of 3D numpy arrays defining
                active regions for each resolution level. Each array should contain
                integer values indicating block/cell activity status.
            sparsity_pattern_origins (List[neon.Index_3d]): List of origin points
                defining the coordinate system offset for each resolution level
            stencil (List[List[int]]): List of 3D offset patterns defining
                computational neighborhoods. Each stencil point is [dx, dy, dz].
                
        Raises:
            Exception: If backend parameter is missing or invalid
            Exception: If grid initialization fails in C++ backend
            
        Example:
            >>> backend = neon.Backend()
            >>> dim = neon.Index_3d(64, 64, 64)
            >>> # Define 2-level hierarchy
            >>> patterns = [fine_pattern, coarse_pattern]  # numpy arrays
            >>> origins = [neon.Index_3d(0,0,0), neon.Index_3d(0,0,0)]
            >>> stencil = [[0,0,0], [1,0,0], [-1,0,0]]  # Simple 3-point stencil
            >>> grid = mGrid(backend, dim, patterns, origins, stencil)
        """

        # Validate required parameters
        if backend is None:
            raise Exception('mGrid: backend parameter is missing')

        # Optional validation: Check if sparsity patterns match domain dimensions
        # This is commented out to allow for flexible grid configurations
        # for sparsity_pattern in sparsity_pattern_list:
        #     if (sparsity_pattern.shape[0] != dim.x or
        #             sparsity_pattern.shape[1] != dim.y or
        #             sparsity_pattern.shape[2] != dim.z):
        #         raise Exception('mGrid: sparsity_pattern\'s shape does not match the dim')

        # Initialize core grid attributes
        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)  # Will be set by C++ constructor
        self.backend = backend
        self.dim = dim
        self.sparsity_pattern_list = sparsity_pattern_list
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.stencil = stencil
        self.num_levels = len(sparsity_pattern_list)

        # Initialize grid with C++ backend
        self._help_load_api()   # Load C++ API functions
        self._help_grid_new()   # Create C++ grid object

    def __del__(self):
        """Destructor - cleanup C++ resources when Python object is garbage collected."""
        if self.handle == 0:
            return
        self._help_grid_delete()

    def _help_load_api(self):
        """
        Load and configure C++ API function pointers from the shared library.
        
        Sets up all the ctypes function signatures for:
        - Grid creation and deletion
        - Dimension queries
        - Span/partition access
        - Domain validation
        - Debug output
        """
        # Initialize Neon gateway and handle management
        self.neon_gate: neon.Gate = neon.Gate()
        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.handle_type = ctypes.c_void_p

        lib = self.neon_gate.lib
        
        # === Grid Lifecycle Management ===
        # Grid creation API - complex signature due to multi-level data
        self.api_new = lib.mGrid_new
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),              # Output: grid handle
                                 self.handle_type,                              # Input: backend handle
                                 ctypes.POINTER(neon.Index_3d),                 # Input: base dimensions
                                 ctypes.c_int,                                  # Input: number of levels
                                 ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # Input: sparsity pattern data pointers
                                 ctypes.POINTER(Index_3d),                      # Input: pattern dimensions
                                 ctypes.POINTER(Index_3d),                      # Input: pattern origins
                                 ctypes.c_int,                                  # Input: stencil size
                                 ctypes.POINTER(ctypes.c_int)]                  # Input: stencil data
        self.api_new.restype = ctypes.c_int

        # Grid deletion API
        self.api_delete = lib.mGrid_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]  # Input: grid handle pointer
        self.api_delete.restype = ctypes.c_int

        # === Grid Query Operations ===
        # Get grid dimensions from C++ object
        self.api_get_dimensions = lib.mGrid_get_dimensions
        self.api_get_dimensions.argtypes = [self.handle_type,                   # Input: grid handle
                                            ctypes.POINTER(neon.Index_3d)]      # Output: dimensions
        self.api_get_dimensions.restype = ctypes.c_int

        # Get execution span for parallel processing
        self.api_get_span = lib.mGrid_get_span
        self.api_get_span.argtypes = [self.handle_type,                         # Input: grid handle
                                      ctypes.c_int,                             # Input: grid level
                                      ctypes.POINTER(bSpan),                    # Output: span object
                                      neon.Execution,                           # Input: execution type
                                      ctypes.c_int,                             # Input: device ID
                                      neon.DataView,                            # Input: data view
                                      ]
        self.api_get_span.restype = ctypes.c_int

        # === Debug and Utility Operations ===
        # Print grid information for debugging
        self.api_print_to_string = lib.mGrid_print_to_string
        self.api_print_to_string.argtypes = [self.handle_type]                  # Input: grid handle
        self.api_print_to_string.restype = ctypes.c_int

        # Check if a point is within the computational domain
        self.api_is_inside_domain = lib.mGrid_is_inside_domain
        self.api_is_inside_domain.argtypes = [self.handle_type,                 # Input: grid handle
                                              ctypes.c_int,                     # Input: grid level
                                              ctypes.POINTER(neon.Index_3d)]    # Input: 3D position
        self.api_is_inside_domain.restype = ctypes.c_bool

    def _help_grid_new(self):
        """
        Create the C++ grid object and initialize the handle.
        
        This method converts Python data structures (numpy arrays, lists) into
        C-compatible formats and calls the C++ grid constructor. It handles
        complex data marshaling for multi-level sparsity patterns and stencils.
        
        Raises:
            Exception: If backend handle is invalid
            Exception: If grid handle is already initialized  
            Exception: If C++ grid creation fails
        """
        def prepare_int32_arrays_and_sizes(np_arrays):
            """
            Convert list of 3D NumPy arrays to C-compatible format.
            
            This nested function prepares sparsity pattern data for transfer to C++.
            It ensures memory layout compatibility and creates the necessary
            pointer arrays and dimension information.
            
            Args:
                np_arrays (List[np.ndarray]): List of 3D numpy arrays with sparsity patterns
                
            Returns:
                tuple: (num_arrays, c_arrays, dims_array, origin_array) where:
                    - num_arrays: Number of arrays in the list
                    - c_arrays: ctypes array of pointers to the data of each array
                    - dims_array: ctypes array of Index_3d with dimensions for each array
                    - origin_array: ctypes array of Index_3d with origins for each level
                    
            Note:
                Each array is made contiguous in memory to ensure proper C++ access.
            """
            # Ensure each array is contiguous in memory for efficient C++ access
            contiguous_arrays = [np.ascontiguousarray(arr) for arr in np_arrays]
            num_arrays = len(contiguous_arrays)

            # Set up ctypes pointer types for 32-bit integer arrays
            Int32P = ctypes.POINTER(ctypes.c_int32)
            ArrayOfPointers = Int32P * num_arrays
            
            # Create array of pointers to numpy data - each pointer points to
            # the start of a flattened 3D array in memory
            c_arrays = ArrayOfPointers(*(arr.ctypes.data_as(Int32P) for arr in contiguous_arrays))

            # Create array of Index_3d objects containing dimensions for each level
            ArraysOfIndex3d = Index_3d * num_arrays
            dims_array = ArraysOfIndex3d()
            for idx, arr in enumerate(contiguous_arrays):
                dims_array[idx] = Index_3d(arr.shape[0], arr.shape[1], arr.shape[2])

            # Create array of origin points for coordinate system offsets
            origin_array_type = Index_3d * len(self.sparsity_pattern_origins)
            origin_array = origin_array_type()
            for idx, origin in enumerate(self.sparsity_pattern_origins):
                origin_array[idx] = origin

            return num_arrays, c_arrays, dims_array, origin_array

        # Validate prerequisites
        if self.backend.backend_handle.value == ctypes.c_void_p(0):
            raise Exception('mGrid: Invalid backend handle')

        if self.handle.value != None:
            raise Exception('mGrid: Grid handle already initialized')

        # Prepare sparsity pattern data for C++ consumption
        num_arrays, c_arrays, dims, origins = prepare_int32_arrays_and_sizes(self.sparsity_pattern_list)
        self.depth = num_arrays

        # Convert stencil patterns to flat C array format
        # Each stencil point [dx, dy, dz] becomes 3 consecutive array elements
        stencil_type = ctypes.c_int * (3 * len(self.stencil))
        stencil_array = stencil_type()
        for s_idx, s in enumerate(self.stencil):
            a_idx = s_idx * 3
            stencil_array[a_idx] = s[0]      # dx offset
            stencil_array[a_idx + 1] = s[1]  # dy offset
            stencil_array[a_idx + 2] = s[2]  # dz offset

        # Call C++ grid constructor with all prepared data
        res = self.api_new(ctypes.pointer(self.handle),    # Output: grid handle
                           self.backend.backend_handle,    # Input: backend handle
                           self.dim,                       # Input: base dimensions
                           self.depth,                     # Input: number of levels
                           c_arrays, dims, origins,        # Input: sparsity pattern data
                           len(self.stencil),              # Input: stencil size
                           stencil_array)                  # Input: stencil data
        if res != 0:
            raise Exception('mGrid: Failed to initialize grid')
        print(f"mGrid initialized with handle {self.handle.value}")

    def _help_grid_delete(self):
        """
        Clean up C++ grid resources.
        
        This method is called automatically by the destructor but can also
        be called manually to free resources earlier.
        
        Raises:
            Exception: If grid deletion fails in C++ backend
        """
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete grid')

    def get_python_dimensions(self):
        """
        Get the grid dimensions as stored in Python.
        
        Returns:
            neon.Index_3d: Base dimensions of the computational domain
        """
        return self.dim

    def get_cpp_dimensions(self):
        """
        Get the grid dimensions from the C++ backend.
        
        This method queries the C++ grid object directly, which may differ
        from the Python-stored dimensions in some edge cases.
        
        Returns:
            neon.Index_3d: Grid dimensions as reported by C++ backend
            
        Raises:
            Exception: If dimension query fails
        """
        cpp_dim = Index_3d(0, 0, 0)
        res = self.neon.lib.mGrid_get_dimensions(self.handle, cpp_dim)
        if res != 0:
            raise Exception('mGrid: Failed to obtain grid dimension')

        return cpp_dim

    def new_field(self,
                  cardinality: ctypes.c_int,
                  dtype,
                  memory_type: neon.MemoryType) -> mField:
        """
        Create a new field associated with this grid.
        
        Fields represent data arrays that are partitioned according to the
        grid's multi-resolution hierarchy. Each field can have multiple
        components (cardinality > 1) and different data types.
        
        Args:
            cardinality (ctypes.c_int): Number of components per grid point
                (1 = scalar field, 3 = vector field, etc.)
            dtype: Python data type for field elements (float, int, etc.)
            memory_type (neon.MemoryType): Memory allocation strategy
                (HOST, DEVICE, or UNIFIED)
                
        Returns:
            mField: New field object configured for this grid
            
        Example:
            >>> scalar_field = grid.new_field(1, float, neon.MemoryType.HOST)
            >>> vector_field = grid.new_field(3, float, neon.MemoryType.DEVICE)
        """
        field = mField(neon_gate=self.neon_gate,
                       grid_handle=self.handle,
                       cardinality=cardinality,
                       memory_type=memory_type,
                       py_grid=self,
                       dtype=dtype
                       )
        return field

    def get_span(self,
                 grid_level: ctypes.c_int,
                 execution: Execution,
                 dev_idx: ctypes.c_int,
                 data_view: DataView) -> bSpan:
        """
        Get an execution span for parallel processing at a specific grid level.
        
        A span represents a portion of the computational domain that can be
        processed independently by a single thread or device. This enables
        data parallelism across the multi-resolution hierarchy.
        
        Args:
            grid_level (ctypes.c_int): Resolution level (0 = finest, higher = coarser)
            execution (Execution): Execution context (HOST or DEVICE)
            dev_idx (ctypes.c_int): Device identifier for GPU execution
            data_view (DataView): Data access pattern (STANDARD or BOUNDARY)
            
        Returns:
            bSpan: Span object defining the iteration space for parallel execution
            
        Raises:
            Exception: If grid handle is invalid
            Exception: If span creation fails
        """
        if self.handle == 0:
            raise Exception('mGrid: Invalid handle')

        span = bSpan()
        res = self.api_get_span(self.handle, grid_level, span, execution, dev_idx, data_view)
        if res != 0:
            raise Exception('Failed to get span')

        # Optional size validation (commented out for performance)
        # cpp_size = self.neon.lib.mGrid_span_size(span)
        # ctypes_size = ctypes.sizeof(span)
        #
        # if cpp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span

    def getProperties(self, grid_level: ctypes.c_int, idx: Index_3d):
        """
        Get data view properties for a specific grid location.
        
        Args:
            grid_level (ctypes.c_int): Resolution level to query
            idx (Index_3d): 3D coordinate position
            
        Returns:
            DataView: Properties of the data at the specified location
        """
        return DataView(self.neon.lib.mGrid_get_properties(self.handle, grid_level, idx))

    def isInsideDomain(self, grid_level: ctypes.c_int, idx: Index_3d):
        """
        Check if a 3D coordinate is within the computational domain.
        
        This method validates whether a given coordinate falls within the
        active region of the grid at a specific resolution level.
        
        Args:
            grid_level (ctypes.c_int): Resolution level to check against
            idx (Index_3d): 3D coordinate position to validate
            
        Returns:
            bool: True if the coordinate is within the domain, False otherwise
            
        Raises:
            Exception: If any coordinate component is negative
        """
        if idx.x < 0 or idx.y < 0 or idx.z < 0:
            raise Exception('can\'t access negative indices in mGrid')
        return self.neon.lib.mGrid_is_inside_domain(self.handle, grid_level, idx)

    def get_backend(self):
        """
        Get the backend configuration used by this grid.
        
        Returns:
            neon.Backend: Backend object managing execution context
        """
        return self.backend

    def get_handle(self):
        """
        Get the C++ object handle.
        
        Returns:
            ctypes.c_void_p: Handle to the underlying C++ grid object
        """
        return self.handle

    def get_name(self):
        """
        Get the grid type name.
        
        Returns:
            str: Grid type identifier ("mGrid")
        """
        return "mGrid"

    def get_num_levels(self):
        """
        Get the number of resolution levels in the hierarchy.
        
        Returns:
            int: Number of levels (depth) in the multi-resolution structure
        """
        return self.num_levels

    def get_dimensions(self):
        """
        Get the base grid dimensions.
        
        Returns:
            neon.Index_3d: Base dimensions of the computational domain
        """
        return self.dim

    def print_info(self):
        """
        Print grid information for debugging.
        
        This method calls the C++ backend to generate detailed information
        about the grid structure, which is useful for debugging and optimization.
        
        Returns:
            int: Status code (0 = success)
            
        Raises:
            Exception: If printing fails
        """
        res = self.api_print_to_string(self.handle)
        if res != 0:
            raise Exception('mGrid: Failed to print grid info')
        return res
