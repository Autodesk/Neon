"""
Dense Grid Implementation for Neon Computing Framework

This module provides the dGrid class, which represents a dense 3D grid
data structure for parallel computing applications. The grid supports
sparsity patterns for memory optimization and various computational
stencils for efficient numerical computations.
"""

import ctypes
from typing import List, Optional, Union
from enum import Enum

import numpy as np
import neon
from neon.dataView import DataView
from neon.execution import Execution
from neon.index_3d import Index_3d
from .dField import dField
from .dSpan import dSpan
from ..backend import Backend


class GridError(Exception):
    """Base exception for grid operations."""
    pass


class InvalidGridHandleError(GridError):
    """Raised when grid handle operations fail."""
    pass


class GridInitializationError(GridError):
    """Raised when grid initialization fails."""
    pass


class SparsityPatternError(GridError):
    """Raised when sparsity patterns are invalid."""
    pass


class dGrid(object):
    """
    Dense Grid (dGrid) class for 3D computational domains.
    
    This class represents a 3D grid structure with a specified sparsity pattern and backend.
    It provides the foundation for dense field computations in the Neon framework, supporting
    various execution backends and data layouts.
    
    The dGrid manages:
    - 3D spatial dimensions and domain boundaries
    - Sparsity patterns for memory optimization
    - Stencil operations for computational kernels
    - Backend-specific optimizations and memory management
    - Field creation and data access patterns
    
    Attributes:
        grid_handle (ctypes.c_void_p): C++ grid object handle
        backend (Backend): Computational backend (CPU, CUDA, etc.)
        dim (Index_3d): Grid dimensions (x, y, z)
        sparsity (np.ndarray): 3D sparsity mask (1=active, 0=inactive)
        stencil (List[List[int]]): Stencil pattern for neighboring cell access
        neon_gate (neon.Gate): Interface to C++ Neon library
        
    Example:
        >>> from neon import Backend, Index_3d
        >>> import numpy as np
        >>> 
        >>> # Create a dense grid with default sparsity
        >>> backend = Backend()
        >>> grid = dGrid(backend=backend, dim=Index_3d(64, 64, 64))
        >>> 
        >>> # Create a sparse grid with custom pattern
        >>> sparsity = np.ones((32, 32, 32))
        >>> sparsity[0:10, 0:10, 0:10] = 0  # Remove corner region
        >>> grid = dGrid(backend=backend, dim=Index_3d(32, 32, 32), sparsity=sparsity)
        >>> 
        >>> # Create fields on the grid
        >>> velocity_field = grid.new_field(cardinality=3, dtype=np.float32)
        >>> pressure_field = grid.new_field(cardinality=1, dtype=np.float64)
    """

    def __init__(
            self, backend: Backend = None,
            dim: Index_3d = Index_3d(10, 10, 10),
            sparsity: np.ndarray = None,
            stencil: List[List[int]] = []
    ):
        """
        Initialize a dense grid with specified parameters.

        Args:
            backend (Backend): Computational backend for execution (CPU, CUDA, etc.).
                              Must not be None.
            dim (Index_3d, optional): Grid dimensions in 3D space. 
                                    Defaults to Index_3d(10, 10, 10).
            sparsity (np.ndarray, optional): 3D boolean mask defining active/inactive cells.
                                           Shape must match dim. If None, creates a fully dense grid.
            stencil (List[List[int]], optional): List of 3D offsets defining the computational
                                               stencil pattern. Each inner list contains [x, y, z]
                                               offsets from the center cell. Defaults to empty list.

        Raises:
            Exception: If backend is None (backend is required).
            Exception: If sparsity array shape doesn't match the specified dimensions.
            Exception: If PyNeon initialization fails.
            
        Note:
            The sparsity array uses the convention: 1 = active cell, 0 = inactive cell.
            Inactive cells are excluded from computations and memory allocation.
        """
        # Validate input parameters
        self._validate_construction_parameters(backend, dim, sparsity, stencil)
        
        # Set default sparsity if not provided
        if sparsity is None:
            sparsity = np.ones((dim.x, dim.y, dim.z))

        # Initialize core grid attributes
        self._handle: ctypes.c_void_p = ctypes.c_void_p(0)  # Will be set by C++ constructor
        self._backend = backend
        self.dim = dim
        self.sparsity = sparsity
        self.stencil = stencil

        # Initialize grid with C++ backend
        self._help_load_api()   # Load C++ API functions
        self._help_grid_new()   # Create C++ grid object

    def __del__(self):
        """Destructor - cleanup C++ resources when Python object is garbage collected."""
        self.cleanup()

    def __enter__(self) -> 'dGrid':
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
                                        backend: Backend,
                                        dim: Index_3d,
                                        sparsity: np.ndarray,
                                        stencil: List[List[int]]) -> None:
        """
        Validate input parameters for grid construction.
        
        Args:
            backend: Backend configuration object
            dim: Grid dimensions
            sparsity: Sparsity pattern array
            stencil: Stencil pattern list
            
        Raises:
            GridInitializationError: If backend is None
            SparsityPatternError: If sparsity dimensions don't match grid dimensions
        """
        if backend is None:
            raise GridInitializationError('Backend parameter is required')
            
        if sparsity is not None:
            if (sparsity.shape[0] != dim.x or 
                sparsity.shape[1] != dim.y or 
                sparsity.shape[2] != dim.z):
                raise SparsityPatternError(
                    f'Sparsity shape {sparsity.shape} does not match grid dimensions '
                    f'({dim.x}, {dim.y}, {dim.z})'
                )

    def _help_load_api(self):
        """
        Load and configure C++ API function bindings.
        
        Initializes the Neon gate interface and sets up ctypes bindings for all
        grid-related C++ functions. This includes setting proper argument types
        and return types for type-safe interfacing with the C++ library.
        
        Raises:
            Exception: If PyNeon initialization fails.
        """
        try:
            self.neon_gate: neon.Gate = neon.Gate()
        except Exception as e:
            self._handle: ctypes.c_uint64 = ctypes.c_void_p(0)
            raise GridInitializationError('Failed to initialize PyNeon: ' + str(e))

        # Get reference to the C++ library object
        lib_obj = self.neon_gate.lib
        
        # Grid creation/destruction API
        self.api_new = getattr(lib_obj, 'dGrid_new')
        self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),  # output grid handle
                                 self.neon_gate.handle_type,                  # backend handle
                                 ctypes.POINTER(neon.Index_3d),               # grid dimensions
                                 ctypes.POINTER(ctypes.c_int),                # sparsity array
                                 ctypes.c_int,                                # stencil size
                                 ctypes.POINTER(ctypes.c_int)]                # stencil data
        self.api_new.restype = ctypes.c_int  # Return code: 0 = success
        
        self.api_delete = getattr(lib_obj, 'dGrid_delete')
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = ctypes.c_int
        
        # Grid property queries
        self.api_get_dimensions = getattr(lib_obj, 'dGrid_get_dimensions')
        self.api_get_dimensions.argtypes = [self.neon_gate.handle_type,
                                            ctypes.POINTER(neon.Index_3d)]
        self.api_get_dimensions.restype = ctypes.c_int
        
        # Span management for iteration and data access
        self.api_get_span = getattr(lib_obj, 'dGrid_get_span')
        self.api_get_span.argtypes = [self.neon_gate.handle_type,
                                      ctypes.POINTER(dSpan),  # output span object
                                      ctypes.c_int,           # execution type (host/device)
                                      ctypes.c_int,           # device id
                                      ctypes.c_int]           # data view (standard/boundary/etc)
        self.api_get_span.restype = ctypes.c_int
        
        self.api_span_size = getattr(lib_obj, 'dGrid_span_size')
        self.api_span_size.argtypes = [ctypes.POINTER(dSpan)]
        self.api_span_size.restype = ctypes.c_int
        
        # Cell property and domain queries
        self.api_get_properties = getattr(lib_obj, 'dGrid_get_properties')
        self.api_get_properties.argtypes = [self.neon_gate.handle_type,
                                            ctypes.POINTER(neon.Index_3d)]
        self.api_get_properties.restype = ctypes.c_int
        
        self.api_is_inside_domain = getattr(lib_obj, 'dGrid_is_inside_domain')
        self.api_is_inside_domain.argtypes = [self.neon_gate.handle_type,
                                              ctypes.POINTER(neon.Index_3d)]
        self.api_is_inside_domain.restype = ctypes.c_bool

    def _help_grid_new(self):
        """
        Create and initialize the underlying C++ grid object.
        
        Converts Python data structures (sparsity, stencil) to C-compatible formats
        and calls the C++ grid constructor. This establishes the grid's memory layout,
        domain decomposition, and computational patterns.
        
        Raises:
            Exception: If grid handle is already initialized.
            Exception: If C++ grid creation fails.
        """
        if self._handle.value != None:  # Ensure the grid handle is uninitialized
            raise GridInitializationError('Grid handle already initialized')

        # Convert stencil list to flat C array format
        # Each stencil point requires 3 integers (x, y, z offsets)
        stencil_type = ctypes.c_int * (3*len(self.stencil))
        stencil_array = stencil_type()
        for s_idx, s in enumerate(self.stencil):
            a_idx = s_idx * 3
            stencil_array[a_idx] = s[0]        # x offset
            stencil_array[a_idx + 1] = s[1]    # y offset
            stencil_array[a_idx + 2] = s[2]    # z offset

        # Convert numpy sparsity array to C pointer
        sparsity_array = self.sparsity.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # Call C++ grid constructor
        res = self.api_new(ctypes.pointer(self._handle),
                           self._backend.backend_handle,
                           self.dim,
                           sparsity_array,
                           len(self.stencil),
                           stencil_array)
        if res != 0:
            raise GridInitializationError('Failed to initialize grid')
        #print(f"dGrid initialized with handle {self._handle.value}")

    def _help_grid_delete(self):
        """
        Clean up and destroy the underlying C++ grid object.
        
        Releases all memory and resources associated with the grid.
        Should only be called once during object destruction.
        
        Raises:
            Exception: If C++ grid deletion fails.
        """
        if self.api_delete(ctypes.pointer(self._handle)) != 0:
            raise GridError('Failed to delete grid')

    def get_python_dimensions(self):
        """
        Get the grid dimensions as stored in Python.
        
        Returns:
            Index_3d: The grid dimensions (x, y, z) as originally specified.
        """
        return self.dim

    def get_cpp_dimensions(self):
        """
        Get the grid dimensions as reported by the C++ backend.
        
        This may differ from Python dimensions due to internal padding,
        alignment, or domain decomposition optimizations.
        
        Returns:
            Index_3d: The actual grid dimensions used by the C++ implementation.
            
        Raises:
            Exception: If unable to query dimensions from C++ backend.
        """
        cpp_dim = Index_3d(0, 0, 0)
        res = self.api_get_dimensions(ctypes.byref(self._handle), cpp_dim)
        if res != 0:
            raise GridError('Failed to obtain grid dimension')

        return cpp_dim

    def new_field(self,
                  cardinality: int,
                  dtype) -> dField:
        """
        Create a new field associated with this grid.
        
        Fields store data values at each active grid point and provide
        computational operations. Multiple fields can be created on the
        same grid with different data types and cardinalities.
        
        Args:
            cardinality (int): Number of components per grid point.
                             1 = scalar field, 3 = vector field, etc.
            dtype: Data type for field values (e.g., np.float32, np.float64).
                  
        Returns:
            dField: A new field object ready for data operations.
            
        Example:
            >>> # Create scalar pressure field
            >>> pressure = grid.new_field(cardinality=1, dtype=np.float64)
            >>> 
            >>> # Create vector velocity field  
            >>> velocity = grid.new_field(cardinality=3, dtype=np.float32)
        """
        cardinality = ctypes.c_int(cardinality)
        field = dField(neon_gate=self.neon_gate,
                       grid_handle=self._handle,
                       cardinality=cardinality,
                       py_grid=self,
                       dtype=dtype)
        return field

    def get_span(self,
                 execution: Execution,
                 dev_idx: int,
                 data_view: DataView) -> dSpan:
        """
        Get a span object for iterating over grid cells.
        
        Spans define iteration patterns and provide access to grid cells
        for computational kernels. They encapsulate domain decomposition,
        ghost cells, and execution-specific optimizations.
        
        Args:
            execution (Execution): Execution type (HOST, DEVICE, etc.).
            dev_idx (int): Device index for multi-device scenarios.
            data_view (DataView): View type (STANDARD, BOUNDARY, INTERNAL, etc.).
            
        Returns:
            dSpan: Span object containing iteration bounds and metadata.
            
        Raises:
            Exception: If grid handle is invalid.
            Exception: If span creation fails.
            Exception: If span size validation fails.
        """
        if self._handle == 0:
            raise InvalidGridHandleError('Invalid grid handle')

        span = dSpan()
        dev_idx_ctypes = ctypes.c_int(dev_idx)
        res = self.api_get_span(self._handle,
                                span,
                                int(execution),
                                dev_idx_ctypes,
                                int(data_view))
        if res != 0:
            raise GridError('Failed to get span')

        # Validate span object integrity
        cpp_size = self.api_span_size(span)
        ctypes_size = ctypes.sizeof(span)

        if cpp_size != ctypes_size:
            raise GridError(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span

    def get_span_type(self):
        """
        Get the span type used by this grid.
        
        Returns:
            type: The dSpan class type for this grid type.
        """
        return dSpan

    def get_properties(self, idx: Index_3d):
        """
        Get the data view properties for a specific grid cell.
        
        Determines the type of cell (internal, boundary, ghost, etc.)
        at the given grid coordinates.
        
        Args:
            idx (Index_3d): Grid coordinates to query.
            
        Returns:
            DataView: Properties of the cell at the given location.
        """
        return DataView(self.api_get_properties(ctypes.byref(self._handle), idx))

    def is_inside_domain(self, idx: Index_3d):
        """
        Check if a grid coordinate is inside the computational domain.
        
        Tests whether the given coordinates correspond to an active
        (non-sparse) cell within the grid boundaries.
        
        Args:
            idx (Index_3d): Grid coordinates to test.
            
        Returns:
            bool: True if the coordinate is inside the domain, False otherwise.
        """
        return self.api_is_inside_domain(ctypes.byref(self._handle), idx)


    # Properties for modern Python interface
    @property
    def backend(self) -> Backend:
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
        return "dGrid"

    @property
    def dimensions(self) -> Index_3d:
        """Grid dimensions."""
        return self.dim

    @property
    def sparsity_pattern(self) -> np.ndarray:
        """Sparsity pattern array (read-only copy)."""
        return self.sparsity.copy()

    @property
    def stencil_pattern(self) -> List[List[int]]:
        """Computational stencil pattern (read-only copy)."""
        return [point.copy() if isinstance(point, list) else list(point) for point in self.stencil]

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
        
        return (f"dGrid(dims=({self.dimensions.x}, {self.dimensions.y}, {self.dimensions.z}), "
                f"backend={backend_name}, "
                f"handle={hex(self._handle.value) if self._handle else 'None'})")

    def __str__(self) -> str:
        """
        User-friendly string representation.
        
        Returns:
            str: Human-readable description of the grid
        """
        return (f"Dense Grid: dimensions ({self.dimensions.x}×{self.dimensions.y}×{self.dimensions.z})")

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
            'grid_type': 'dGrid',
            'dimensions': (self.dimensions.x, self.dimensions.y, self.dimensions.z),
            'stencil_size': len(self.stencil),
            'sparsity_pattern_memory_bytes': self.sparsity.nbytes,
            'sparsity_info': {
                'shape': self.sparsity.shape,
                'dtype': str(self.sparsity.dtype),
                'total_elements': int(self.sparsity.size),
                'active_elements': int(np.count_nonzero(self.sparsity)),
            }
        }
        
        if info['sparsity_info']['total_elements'] > 0:
            info['sparsity_info']['sparsity_ratio'] = (
                info['sparsity_info']['active_elements'] / info['sparsity_info']['total_elements']
            )
            info['sparsity_info']['compression_ratio'] = 1.0 - info['sparsity_info']['sparsity_ratio']
        
        return info

    def get_grid_statistics(self) -> dict:
        """
        Get statistical information about the grid structure.
        
        Returns:
            dict: Dictionary containing grid statistics
        """
        stats = {
            'grid_type': 'dGrid',
            'dimensions': (self.dimensions.x, self.dimensions.y, self.dimensions.z),
            'total_stencil_points': len(self.stencil),
            'sparsity_statistics': {}
        }
        
        active_count = int(np.count_nonzero(self.sparsity))
        total_count = int(self.sparsity.size)
        sparsity_ratio = active_count / total_count if total_count > 0 else 0.0
        
        stats['sparsity_statistics'] = {
            'shape': self.sparsity.shape,
            'total_elements': total_count,
            'active_elements': active_count,
            'sparsity_ratio': sparsity_ratio,
            'compression_ratio': 1.0 - sparsity_ratio
        }
        
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
            # Check handle validity
            if not self._handle or self._handle.value == 0:
                raise GridError("Invalid C++ handle")
            
            # Check dimensions
            if self.dimensions.x <= 0 or self.dimensions.y <= 0 or self.dimensions.z <= 0:
                raise GridError("Invalid grid dimensions")
            
            # Check sparsity pattern
            if self.sparsity is None:
                raise GridError("Sparsity pattern is None")
            if self.sparsity.ndim != 3:
                raise GridError("Sparsity pattern is not 3D")
            if self.sparsity.size == 0:
                raise GridError("Sparsity pattern is empty")
            
            # Check sparsity dimensions match grid dimensions
            if (self.sparsity.shape[0] != self.dimensions.x or
                self.sparsity.shape[1] != self.dimensions.y or
                self.sparsity.shape[2] != self.dimensions.z):
                raise GridError("Sparsity pattern dimensions don't match grid dimensions")
            
            # Check stencil
            if not self.stencil:
                raise GridError("Grid must have a stencil pattern")
            
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
