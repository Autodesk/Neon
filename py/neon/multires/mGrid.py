"""
Multi-resolution Grid Implementation for Neon Computing Framework

This module provides the mGrid class, which represents a multi-resolution
hierarchical grid data structure for parallel computing applications. The grid
supports adaptive mesh refinement patterns and enables efficient computation
across multiple resolution levels.

"""

import ctypes
from typing import List, Optional, Union
from enum import Enum

import neon
from .mField import mField
from neon.execution import Execution
from neon.dataView import DataView
from ..block.bSpan import bSpan
from neon.index_3d import Index_3d
import numpy as np


class ExecutionContext(Enum):
    """Enumeration for execution contexts in parallel computing."""
    HOST = "HOST"
    DEVICE = "DEVICE"


class GridError(Exception):
    """Base exception for grid operations."""
    pass


class InvalidGridLevelError(GridError):
    """Raised when grid level is out of bounds."""
    
    def __init__(self, level: int, max_levels: int):
        self.level = level
        self.max_levels = max_levels
        super().__init__(f"Grid level {level} is out of bounds [0, {max_levels})")


class DomainBoundsError(GridError):
    """Raised when coordinates are outside domain boundaries."""
    
    def __init__(self, idx: Index_3d, message: str = "Coordinates outside domain"):
        self.idx = idx
        super().__init__(f"{message}: ({idx.x}, {idx.y}, {idx.z})")


class GridInitializationError(GridError):
    """Raised when grid initialization fails."""
    pass


class InvalidBackendError(GridError):
    """Raised when backend configuration is invalid."""
    pass


class SparsityPatternError(GridError):
    """Raised when sparsity patterns are invalid."""
    pass


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
        self._validate_construction_parameters(backend, dim, sparsity_pattern_list, 
                                             sparsity_pattern_origins, stencil)

        # Initialize core grid attributes
        self._handle: ctypes.c_void_p = ctypes.c_void_p(0)  # Will be set by C++ constructor
        self._backend = backend
        self.dim = dim
        self.sparsity_pattern_list = sparsity_pattern_list
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.stencil = stencil
        # Dense construction path: no sparse active-voxel coordinate lists.
        self.active_voxels_list = None
        self._num_levels = len(sparsity_pattern_list)

        # Initialize grid with C++ backend
        self._help_load_api()   # Load C++ API functions
        self._help_grid_new()   # Create C++ grid object

    @classmethod
    def from_active_voxels(cls,
                           backend: neon.Backend,
                           dim,
                           active_voxels_list: List[np.ndarray],
                           sparsity_pattern_origins: List[neon.Index_3d],
                           stencil: List[List[int]]) -> 'mGrid':
        """
        Create a multi-resolution grid from sparse per-level active voxels.

        This is the sparse counterpart of the (dense) constructor. Instead of a
        dense 3D mask per level it takes, for each level, an (N, 3) array of the
        coordinates of the active voxels. This avoids materialising a dense
        array per level, which is prohibitive for large sparse domains (the
        level-0 dense mask alone can be terabytes).

        Coordinates are expressed in each level's local (scaled) index space,
        i.e. the same space as the indices of the dense ``sparsity_pattern_list``
        arrays. Concretely, a base-index-space voxel ``idx`` is active at level
        ``l`` iff ``(idx >> l) - origin[l]`` is a registered coordinate. With the
        default origin of (0, 0, 0) this is simply the level-``l`` grid
        coordinate of the voxel. The resulting grid is identical to the one the
        dense path would build from equivalent masks.

        Args:
            backend (neon.Backend): Neon backend configuration.
            dim (neon.Index_3d): Base dimensions of the computational domain.
            active_voxels_list (List[np.ndarray]): One (N_l, 3) integer array per
                level listing the active voxel coordinates for that level. An
                empty (0, 3) array denotes a level with no active voxels.
            sparsity_pattern_origins (List[neon.Index_3d]): One origin per level.
            stencil (List[List[int]]): Stencil offsets, each [dx, dy, dz].

        Returns:
            mGrid: A new grid constructed via the sparse ingestion path.

        Example:
            >>> coords0 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
            >>> coords1 = np.array([[0, 0, 0]], dtype=np.int32)
            >>> grid = mGrid.from_active_voxels(
            ...     backend, neon.Index_3d(64, 64, 64),
            ...     [coords0, coords1],
            ...     [neon.Index_3d(0, 0, 0), neon.Index_3d(0, 0, 0)],
            ...     [[0, 0, 0], [1, 0, 0]])
        """
        self = cls.__new__(cls)

        # Normalise each level to a contiguous (N, 3) int32 array.
        normalized = [np.ascontiguousarray(np.asarray(a, dtype=np.int32)).reshape(-1, 3)
                      for a in active_voxels_list]

        self._validate_sparse_construction_parameters(
            backend, dim, normalized, sparsity_pattern_origins, stencil)

        self._handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self._backend = backend
        self.dim = dim
        # Sparse construction path: no dense masks are retained.
        self.sparsity_pattern_list = None
        self.active_voxels_list = normalized
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.stencil = stencil
        self._num_levels = len(normalized)

        self._help_load_api()
        self._help_grid_new_sparse()
        return self

    def __del__(self):
        """Destructor - cleanup C++ resources when Python object is garbage collected."""
        self.cleanup()

    def __enter__(self) -> 'mGrid':
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit point with automatic cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self) -> None:
        """
        Explicit cleanup method for C++ resources.
        
        This method is idempotent and can be called multiple times safely.
        It's automatically called by the context manager and destructor.
        """
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        
        if hasattr(self, '_handle') and self._handle and self._handle != 0:
            self._help_grid_delete()
        
        self._cleaned = True

    def _validate_construction_parameters(self, 
                                        backend: neon.Backend,
                                        dim: neon.Index_3d,
                                        sparsity_pattern_list: List[np.ndarray],
                                        sparsity_pattern_origins: List[neon.Index_3d],
                                        stencil: List[List[int]]) -> None:
        """
        Validate all construction parameters.
        
        Args:
            backend: Neon backend configuration
            dim: Base grid dimensions
            sparsity_pattern_list: List of sparsity patterns
            sparsity_pattern_origins: List of origin points
            stencil: Stencil pattern definition
            
        Raises:
            InvalidBackendError: If backend is invalid
            SparsityPatternError: If sparsity patterns are invalid
            ValueError: If other parameters are invalid
        """
        # Validate backend
        if backend is None:
            raise InvalidBackendError("Backend parameter is required")
        
        # Validate dimensions
        if dim is None:
            raise ValueError("Grid dimensions are required")
        if dim.x <= 0 or dim.y <= 0 or dim.z <= 0:
            raise ValueError(f"Grid dimensions must be positive: ({dim.x}, {dim.y}, {dim.z})")
        
        # Validate sparsity patterns
        if not sparsity_pattern_list:
            raise SparsityPatternError("At least one sparsity pattern is required")
        
        if len(sparsity_pattern_list) != len(sparsity_pattern_origins):
            raise SparsityPatternError(
                f"Number of sparsity patterns ({len(sparsity_pattern_list)}) "
                f"must match number of origins ({len(sparsity_pattern_origins)})"
            )
        
        for i, pattern in enumerate(sparsity_pattern_list):
            if pattern is None:
                raise SparsityPatternError(f"Sparsity pattern at level {i} is None")
            if pattern.ndim != 3:
                raise SparsityPatternError(f"Sparsity pattern at level {i} must be 3D, got {pattern.ndim}D")
            if pattern.size == 0:
                raise SparsityPatternError(f"Sparsity pattern at level {i} is empty")
        
        # Validate stencil
        if not stencil:
            raise ValueError("Stencil pattern is required")
        
        for i, point in enumerate(stencil):
            if len(point) != 3:
                raise ValueError(f"Stencil point {i} must have 3 coordinates, got {len(point)}")

    def _validate_sparse_construction_parameters(self,
                                                 backend: neon.Backend,
                                                 dim: neon.Index_3d,
                                                 active_voxels_list: List[np.ndarray],
                                                 sparsity_pattern_origins: List[neon.Index_3d],
                                                 stencil: List[List[int]]) -> None:
        """
        Validate parameters for the sparse (active-voxel) construction path.

        Args:
            backend: Neon backend configuration.
            dim: Base grid dimensions.
            active_voxels_list: List of (N_l, 3) integer coordinate arrays.
            sparsity_pattern_origins: List of origin points.
            stencil: Stencil pattern definition.

        Raises:
            InvalidBackendError: If backend is invalid.
            SparsityPatternError: If the coordinate arrays are invalid.
            ValueError: If other parameters are invalid.
        """
        if backend is None:
            raise InvalidBackendError("Backend parameter is required")

        if dim is None:
            raise ValueError("Grid dimensions are required")
        if dim.x <= 0 or dim.y <= 0 or dim.z <= 0:
            raise ValueError(f"Grid dimensions must be positive: ({dim.x}, {dim.y}, {dim.z})")

        if not active_voxels_list:
            raise SparsityPatternError("At least one level of active voxels is required")

        if len(active_voxels_list) != len(sparsity_pattern_origins):
            raise SparsityPatternError(
                f"Number of active-voxel levels ({len(active_voxels_list)}) "
                f"must match number of origins ({len(sparsity_pattern_origins)})"
            )

        for i, coords in enumerate(active_voxels_list):
            if coords is None:
                raise SparsityPatternError(f"Active voxels at level {i} is None")
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise SparsityPatternError(
                    f"Active voxels at level {i} must have shape (N, 3), got {coords.shape}")
            if coords.size and coords.min() < 0:
                # Only non-negative coordinates can ever match a base-index-space
                # voxel, so negative coordinates almost certainly indicate a bug.
                raise SparsityPatternError(
                    f"Active voxels at level {i} contain negative coordinates")

        if not stencil:
            raise ValueError("Stencil pattern is required")

        for i, point in enumerate(stencil):
            if len(point) != 3:
                raise ValueError(f"Stencil point {i} must have 3 coordinates, got {len(point)}")

    def _validate_grid_level(self, level: int) -> None:
        """
        Validate grid level parameter.
        
        Args:
            level: Grid level to validate
            
        Raises:
            TypeError: If level is not an integer
            InvalidGridLevelError: If level is out of bounds
        """
        if not isinstance(level, int):
            raise TypeError(f"Grid level must be integer, got {type(level)}")
        if not 0 <= level < self.num_levels:
            raise InvalidGridLevelError(level, self.num_levels)

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
        self._handle: ctypes.c_void_p = ctypes.c_void_p(0)
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

        # Sparse grid creation API - accepts per-level active-voxel coordinate lists
        self.api_new_sparse = lib.mGrid_new_sparse
        self.api_new_sparse.argtypes = [ctypes.POINTER(self.handle_type),                # Output: grid handle
                                        self.handle_type,                                # Input: backend handle
                                        ctypes.POINTER(neon.Index_3d),                   # Input: base dimensions
                                        ctypes.c_int,                                    # Input: number of levels
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),  # Input: per-level coord arrays
                                        ctypes.POINTER(ctypes.c_int32),                  # Input: per-level voxel counts
                                        ctypes.POINTER(Index_3d),                        # Input: per-level origins
                                        ctypes.c_int,                                    # Input: stencil size
                                        ctypes.POINTER(ctypes.c_int)]                    # Input: stencil data
        self.api_new_sparse.restype = ctypes.c_int

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
                                      ctypes.c_int,                             # Input: execution type
                                      ctypes.c_int,                             # Input: device ID
                                      ctypes.c_int,                             # Input: data view
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
            raise InvalidBackendError('Backend handle is invalid')

        if self._handle.value != None:
            raise GridInitializationError('Grid handle already initialized')

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
        res = self.api_new(ctypes.pointer(self._handle),   # Output: grid handle
                           self.backend.backend_handle,    # Input: backend handle
                           self.dim,                       # Input: base dimensions
                           self.depth,                     # Input: number of levels
                           c_arrays, dims, origins,        # Input: sparsity pattern data
                           len(self.stencil),              # Input: stencil size
                           stencil_array)                  # Input: stencil data
        if res != 0:
            raise GridInitializationError(f'Failed to initialize grid (error code: {res})')
        
        # Update public handle for backward compatibility
        from ..logging import logger
        logger.debug(f"mGrid initialized with handle {self._handle.value}")

    def _help_grid_new_sparse(self):
        """
        Create the C++ grid object from sparse per-level active voxels.

        This is the sparse counterpart of :meth:`_help_grid_new`. It passes, for
        each level, a flat int32 coordinate array plus the number of voxels to
        the ``mGrid_new_sparse`` C entry point, which builds a set-backed
        activation predicate per level. No dense mask is allocated.

        Raises:
            InvalidBackendError: If the backend handle is invalid.
            GridInitializationError: If the grid handle is already initialized or
                C++ grid creation fails.
        """
        if self.backend.backend_handle.value == ctypes.c_void_p(0):
            raise InvalidBackendError('Backend handle is invalid')

        if self._handle.value != None:
            raise GridInitializationError('Grid handle already initialized')

        num_levels = len(self.active_voxels_list)

        # Keep references to the contiguous arrays alive for the duration of the
        # C call so their data pointers remain valid.
        self._coord_buffers = [np.ascontiguousarray(arr, dtype=np.int32).reshape(-1, 3)
                               for arr in self.active_voxels_list]

        Int32P = ctypes.POINTER(ctypes.c_int32)
        ArrayOfPointers = Int32P * num_levels
        c_arrays = ArrayOfPointers(*(arr.ctypes.data_as(Int32P) for arr in self._coord_buffers))

        counts_type = ctypes.c_int32 * num_levels
        counts = counts_type(*(arr.shape[0] for arr in self._coord_buffers))

        origin_array_type = Index_3d * len(self.sparsity_pattern_origins)
        origins = origin_array_type()
        for idx, origin in enumerate(self.sparsity_pattern_origins):
            origins[idx] = origin

        self.depth = num_levels

        stencil_type = ctypes.c_int * (3 * len(self.stencil))
        stencil_array = stencil_type()
        for s_idx, s in enumerate(self.stencil):
            a_idx = s_idx * 3
            stencil_array[a_idx] = s[0]
            stencil_array[a_idx + 1] = s[1]
            stencil_array[a_idx + 2] = s[2]

        res = self.api_new_sparse(ctypes.pointer(self._handle),
                                  self.backend.backend_handle,
                                  self.dim,
                                  self.depth,
                                  c_arrays, counts, origins,
                                  len(self.stencil),
                                  stencil_array)
        if res != 0:
            raise GridInitializationError(f'Failed to initialize sparse grid (error code: {res})')

        from ..logging import logger
        logger.debug(f"mGrid (sparse) initialized with handle {self._handle.value}")

    def _help_grid_delete(self):
        """
        Clean up C++ grid resources.
        
        This method is called automatically by the destructor but can also
        be called manually to free resources earlier.
        
        Raises:
            Exception: If grid deletion fails in C++ backend
        """
        res = self.api_delete(ctypes.pointer(self._handle))
        if res != 0:
            raise GridError(f'Failed to delete grid (error code: {res})')

    def get_python_dimensions(self) -> neon.Index_3d:
        """
        Get the grid dimensions as stored in Python.
        
        Returns:
            neon.Index_3d: Base dimensions of the computational domain
        """
        return self.dim

    def get_cpp_dimensions(self) -> Index_3d:
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
        res = self.neon.lib.mGrid_get_dimensions(self._handle, cpp_dim)
        if res != 0:
            raise GridError(f'Failed to obtain grid dimensions (error code: {res})')

        return cpp_dim

    def new_field(self,
                  cardinality: int,
                  dtype,
                  memory_type: neon.MemoryType) -> mField:
        """
        Create a new field associated with this grid.
        
        Fields represent data arrays that are partitioned according to the
        grid's multi-resolution hierarchy. Each field can have multiple
        components (cardinality > 1) and different data types.
        
        Args:
            cardinality (int): Number of components per grid point
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
                       grid_handle=self._handle,
                       cardinality=cardinality,
                       memory_type=memory_type,
                       py_grid=self,
                       dtype=dtype
                       )
        return field

    def get_span(self,
                 grid_level: int,
                 execution: Union[Execution, ExecutionContext],
                 dev_idx: int = 0,
                 data_view: Optional[DataView] = None) -> bSpan:
        """
        Get an execution span for parallel processing at a specific grid level.
        
        A span represents a portion of the computational domain that can be
        processed independently by a single thread or device. This enables
        data parallelism across the multi-resolution hierarchy.
        
        Args:
            grid_level (int): Resolution level (0 = finest, higher = coarser)
            execution (Union[Execution, ExecutionContext]): Execution context (HOST or DEVICE)
            dev_idx (int): Device identifier for GPU execution (default: 0)
            data_view (Optional[DataView]): Data access pattern (default: None for STANDARD)
            
        Returns:
            bSpan: Span object defining the iteration space for parallel execution
            
        Raises:
            InvalidGridLevelError: If grid level is out of bounds
            GridError: If grid handle is invalid or span creation fails
        """
        # Validate inputs
        self._validate_grid_level(grid_level)
        
        if self._handle == 0:
            raise GridError('Grid handle is invalid')
        
        # Handle execution context conversion if needed
        if isinstance(execution, ExecutionContext):
            execution_value = 0 if execution == ExecutionContext.DEVICE else 1
        else:
            execution_value = int(execution)
        
        # Use default data view if not provided
        if data_view is None:
            data_view = DataView.standard()

        span = bSpan()
        res = self.api_get_span(self._handle, grid_level, span, execution_value, dev_idx, int(data_view))
        if res != 0:
            raise GridError(f'Failed to get span for level {grid_level} (error code: {res})')

        # Optional size validation (commented out for performance)
        # cpp_size = self.neon.lib.mGrid_span_size(span)
        # ctypes_size = ctypes.sizeof(span)
        #
        # if cpp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span

    def get_properties(self, grid_level: int, idx: Index_3d) -> DataView:
        """
        Get data view properties for a specific grid location.
        
        Args:
            grid_level (int): Resolution level to query
            idx (Index_3d): 3D coordinate position
            
        Returns:
            DataView: Properties of the data at the specified location
            
        Raises:
            InvalidGridLevelError: If grid level is out of bounds
        """
        # Validate inputs
        self._validate_grid_level(grid_level)
        
        return DataView(self.neon.lib.mGrid_get_properties(self._handle, grid_level, idx))

    def is_inside_domain(self, grid_level: int, idx: Index_3d) -> bool:
        """
        Check if a 3D coordinate is within the computational domain.
        
        This method validates whether a given coordinate falls within the
        active region of the grid at a specific resolution level.
        
        Args:
            grid_level (int): Resolution level to check against
            idx (Index_3d): 3D coordinate position to validate
            
        Returns:
            bool: True if the coordinate is within the domain, False otherwise
            
        Raises:
            InvalidGridLevelError: If grid level is out of bounds
            DomainBoundsError: If any coordinate component is negative
        """
        # Validate inputs
        self._validate_grid_level(grid_level)
        
        if idx.x < 0 or idx.y < 0 or idx.z < 0:
            raise DomainBoundsError(idx, "Negative indices are not allowed")

        return self.api_is_inside_domain(self._handle, grid_level, idx)

    @property
    def backend(self) -> neon.Backend:
        """Backend configuration used by this grid."""
        return self._backend

    @property
    def handle(self) -> ctypes.c_void_p:
        """C++ object handle (read-only)."""
        return self._handle

    @handle.setter
    def handle(self, value: ctypes.c_void_p) -> None:
        """Set the handle (for backward compatibility only)."""
        self._handle = value

    @property
    def name(self) -> str:
        """Grid type name identifier."""
        return "mGrid"

    @property
    def num_levels(self) -> int:
        """Number of resolution levels in the hierarchy."""
        return self._num_levels

    @property
    def is_sparse(self) -> bool:
        """True if this grid was built from sparse active-voxel coordinate lists."""
        return self.sparsity_pattern_list is None

    @property
    def dimensions(self) -> neon.Index_3d:
        """Base grid dimensions."""
        return self.dim

    @property
    def stencil_pattern(self) -> List[List[int]]:
        """Computational stencil pattern (read-only copy)."""
        return [point.copy() for point in self.stencil]

    def print_info(self) -> int:
        """
        Print grid information for debugging.
        
        This method calls the C++ backend to generate detailed information
        about the grid structure, which is useful for debugging and optimization.
        
        Returns:
            int: Status code (0 = success)
            
        Raises:
            GridError: If printing fails
        """
        res = self.api_print_to_string(self._handle)
        if res != 0:
            raise GridError(f'Failed to print grid info (error code: {res})')
        return res

    # Enhanced Debugging and Introspection Methods
    def __repr__(self) -> str:
        """
        Detailed string representation for debugging.
        
        Returns:
            str: Comprehensive representation showing key grid properties
        """
        try:
            backend_name = getattr(self._backend, 'get_name', lambda: 'Unknown')()
        except:
            backend_name = 'Unknown'
        
        return (f"mGrid(levels={self.num_levels}, "
                f"dims=({self.dimensions.x}, {self.dimensions.y}, {self.dimensions.z}), "
                f"backend={backend_name}, "
                f"handle={hex(self._handle.value) if self._handle else 'None'})")

    def __str__(self) -> str:
        """
        User-friendly string representation.
        
        Returns:
            str: Human-readable description of the grid
        """
        return (f"Multi-resolution Grid: {self.num_levels} levels, "
                f"dimensions ({self.dimensions.x}×{self.dimensions.y}×{self.dimensions.z})")

    def get_memory_info(self) -> dict:
        """
        Get memory usage information for the grid.
        
        Returns:
            dict: Dictionary containing memory usage statistics
            
        Note:
            This provides estimates based on grid structure.
            Actual C++ memory usage may differ.
        """
        info = {
            'num_levels': self.num_levels,
            'base_dimensions': (self.dimensions.x, self.dimensions.y, self.dimensions.z),
            'stencil_size': len(self.stencil),
            'estimated_pattern_memory_bytes': 0,
            'patterns_per_level': []
        }

        total_pattern_bytes = 0
        if self.is_sparse:
            for i, coords in enumerate(self.active_voxels_list):
                pattern_bytes = coords.nbytes
                total_pattern_bytes += pattern_bytes
                info['patterns_per_level'].append({
                    'level': i,
                    'shape': coords.shape,
                    'dtype': str(coords.dtype),
                    'size_bytes': pattern_bytes,
                    'active_elements': int(coords.shape[0]),
                    'representation': 'sparse'
                })
        else:
            for i, pattern in enumerate(self.sparsity_pattern_list):
                pattern_bytes = pattern.nbytes
                total_pattern_bytes += pattern_bytes
                info['patterns_per_level'].append({
                    'level': i,
                    'shape': pattern.shape,
                    'dtype': str(pattern.dtype),
                    'size_bytes': pattern_bytes,
                    'active_elements': int(np.count_nonzero(pattern)),
                    'representation': 'dense'
                })

        info['estimated_pattern_memory_bytes'] = total_pattern_bytes
        return info

    def get_grid_statistics(self) -> dict:
        """
        Get statistical information about the grid structure.
        
        Returns:
            dict: Dictionary containing grid statistics
        """
        stats = {
            'num_levels': self.num_levels,
            'total_stencil_points': len(self.stencil),
            'levels_info': []
        }

        if self.is_sparse:
            for i, (coords, origin) in enumerate(zip(self.active_voxels_list,
                                                     self.sparsity_pattern_origins)):
                level_info = {
                    'level': i,
                    'origin': (origin.x, origin.y, origin.z),
                    'shape': coords.shape,
                    'active_elements': int(coords.shape[0]),
                    'representation': 'sparse'
                }
                stats['levels_info'].append(level_info)
            return stats

        for i, (pattern, origin) in enumerate(zip(self.sparsity_pattern_list,
                                                  self.sparsity_pattern_origins)):
            active_count = int(np.count_nonzero(pattern))
            total_count = int(pattern.size)
            sparsity_ratio = active_count / total_count if total_count > 0 else 0.0

            level_info = {
                'level': i,
                'origin': (origin.x, origin.y, origin.z),
                'shape': pattern.shape,
                'total_elements': total_count,
                'active_elements': active_count,
                'sparsity_ratio': sparsity_ratio,
                'compression_ratio': 1.0 - sparsity_ratio,
                'representation': 'dense'
            }
            stats['levels_info'].append(level_info)

        return stats

    def validate_integrity(self) -> bool:
        """
        Validate the integrity of the grid structure.
        
        Returns:
            bool: True if grid structure is valid, False otherwise
            
        Raises:
            GridError: If critical integrity issues are found
        """
        try:
            # Check basic structure
            if self.num_levels == 0:
                raise GridError("Grid must have at least one level")

            levels = self.active_voxels_list if self.is_sparse else self.sparsity_pattern_list
            if len(levels) != len(self.sparsity_pattern_origins):
                raise GridError("Mismatch between patterns and origins")
            
            if not self.stencil:
                raise GridError("Grid must have a stencil pattern")
            
            # Check handle validity
            if not self._handle or self._handle.value == 0:
                raise GridError("Invalid C++ handle")
            
            # Check dimensions
            if self.dimensions.x <= 0 or self.dimensions.y <= 0 or self.dimensions.z <= 0:
                raise GridError("Invalid grid dimensions")

            # Check each level's sparsity representation
            if self.is_sparse:
                for i, coords in enumerate(self.active_voxels_list):
                    if coords is None:
                        raise GridError(f"Active voxels at level {i} is None")
                    if coords.ndim != 2 or coords.shape[1] != 3:
                        raise GridError(f"Active voxels at level {i} is not (N, 3)")
            else:
                for i, pattern in enumerate(self.sparsity_pattern_list):
                    if pattern is None:
                        raise GridError(f"Pattern at level {i} is None")
                    if pattern.ndim != 3:
                        raise GridError(f"Pattern at level {i} is not 3D")
                    if pattern.size == 0:
                        raise GridError(f"Pattern at level {i} is empty")
            
            # Check stencil
            for i, point in enumerate(self.stencil):
                if len(point) != 3:
                    raise GridError(f"Stencil point {i} does not have 3 coordinates")
            
            return True
            
        except GridError:
            raise
        except Exception as e:
            raise GridError(f"Integrity validation failed: {e}")

    def get_debug_info(self) -> dict:
        """
        Get comprehensive debugging information.
        
        Returns:
            dict: Dictionary containing all available debug information
        """
        debug_info = {
            'grid_type': self.name,
            'handle_value': hex(self._handle.value) if self._handle else 'None',
            'is_cleaned': getattr(self, '_cleaned', False),
            'memory_info': self.get_memory_info(),
            'statistics': self.get_grid_statistics(),
            'stencil_pattern': self.stencil_pattern,
            'backend_info': {
                'type': type(self._backend).__name__,
                'handle_value': hex(self._backend.backend_handle.value) if hasattr(self._backend, 'backend_handle') else 'Unknown'
            }
        }
        
        return debug_info


class mGridSparseBuilder(object):
    """
    Incremental builder for a sparse multi-resolution :class:`mGrid`.

    This lets callers register active voxels level by level - either one at a
    time via :meth:`register_voxel` or in bulk via :meth:`register_voxels` - and
    then materialise the grid with :meth:`build`. Registered coordinates are
    accumulated on the Python side and flushed to Neon in a single call, so the
    dense per-level mask is never allocated.

    Coordinates use the same convention as :meth:`mGrid.from_active_voxels`:
    each voxel is given in its level's local (scaled) index space (with the
    default origin of (0, 0, 0) this is simply the level's grid coordinate).

    Example:
        >>> b = mGridSparseBuilder(backend, neon.Index_3d(64, 64, 64),
        ...                        num_levels=2, stencil=[[0, 0, 0], [1, 0, 0]])
        >>> b.register_voxel(0, 0, 0, 0)
        >>> b.register_voxels(1, np.array([[0, 0, 0], [1, 0, 0]]))
        >>> grid = b.build()
    """

    def __init__(self,
                 backend: neon.Backend,
                 dim,
                 num_levels: int,
                 stencil: List[List[int]],
                 origins: Optional[List[neon.Index_3d]] = None):
        """
        Args:
            backend (neon.Backend): Neon backend configuration.
            dim (neon.Index_3d): Base dimensions of the computational domain.
            num_levels (int): Number of resolution levels.
            stencil (List[List[int]]): Stencil offsets, each [dx, dy, dz].
            origins (Optional[List[neon.Index_3d]]): One origin per level.
                Defaults to (0, 0, 0) for every level.
        """
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")

        self.backend = backend
        self.dim = dim
        self.num_levels = num_levels
        self.stencil = stencil
        self.origins = origins if origins is not None else [Index_3d(0, 0, 0)] * num_levels
        # Per-level accumulators; entries may be (x, y, z) tuples or (N, 3) arrays.
        self._levels: List[list] = [[] for _ in range(num_levels)]

    def _check_level(self, level: int) -> None:
        if not isinstance(level, int):
            raise TypeError(f"level must be an integer, got {type(level)}")
        if not 0 <= level < self.num_levels:
            raise InvalidGridLevelError(level, self.num_levels)

    def register_voxel(self, level: int, x: int, y: int, z: int) -> 'mGridSparseBuilder':
        """Register a single active voxel at ``level``. Returns self for chaining."""
        self._check_level(level)
        self._levels[level].append((int(x), int(y), int(z)))
        return self

    def register_voxels(self, level: int, coords) -> 'mGridSparseBuilder':
        """
        Register many active voxels at ``level``.

        Args:
            level (int): Resolution level.
            coords: Array-like of shape (N, 3) of integer coordinates.
        """
        self._check_level(level)
        arr = np.ascontiguousarray(np.asarray(coords, dtype=np.int32)).reshape(-1, 3)
        self._levels[level].append(arr)
        return self

    def num_registered(self, level: int) -> int:
        """Number of voxels registered so far at ``level``."""
        self._check_level(level)
        total = 0
        for chunk in self._levels[level]:
            total += 1 if isinstance(chunk, tuple) else chunk.shape[0]
        return total

    def _collect_level(self, entries: list) -> np.ndarray:
        if not entries:
            return np.empty((0, 3), dtype=np.int32)
        arrays = []
        tuples = []
        for chunk in entries:
            if isinstance(chunk, tuple):
                tuples.append(chunk)
            else:
                arrays.append(chunk)
        if tuples:
            arrays.append(np.asarray(tuples, dtype=np.int32).reshape(-1, 3))
        return np.ascontiguousarray(np.vstack(arrays)).astype(np.int32, copy=False)

    def build(self) -> mGrid:
        """Materialise the accumulated active voxels into an :class:`mGrid`."""
        active_voxels_list = [self._collect_level(level) for level in self._levels]
        return mGrid.from_active_voxels(
            self.backend,
            self.dim,
            active_voxels_list,
            self.origins,
            self.stencil,
        )
