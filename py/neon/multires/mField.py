"""
Multi-resolution Field Implementation for Neon Computing Framework

This module provides the mField class, which represents a multi-resolution field
data structure for parallel computing applications. The field supports various
data types and memory configurations for both CPU and GPU execution.

"""

import ctypes
import warnings
from typing import Any, Optional, Dict, Union
import numpy as np

import neon
import neon.multires.mPartition


class FieldError(Exception):
    """Base exception for field operations."""
    pass


class InvalidFieldHandleError(FieldError):
    """Raised when field handle is invalid."""
    pass


class FieldInitializationError(FieldError):
    """Raised when field initialization fails."""
    pass


class PartitionError(FieldError):
    """Raised when partition operations fail."""
    pass


class DataTransferError(FieldError):
    """Raised when data transfer operations fail."""
    pass


class mField(object):
    """
    Multi-resolution Field for distributed and parallel computing.
    
    The mField class represents a multi-dimensional data field that can be
    partitioned across multiple resolution levels. It provides a Python interface
    to the underlying C++ implementation for high-performance computing applications.
    
    This class supports:
    - Multiple data types (int, float, double, etc.)
    - Different memory configurations (host, device, unified)
    - Multi-resolution data structures
    - Parallel execution across multiple devices
    - Data import/export capabilities (VTI format)
    
    Attributes:
        dtype: Python data type of the field elements
        neon_gate: Gateway to the Neon C++ library
        handle: C++ object handle for the field
        grid_handle: Handle to the parent grid object
        cardinality: Number of components per field element
        memory_type: Memory allocation type (host/device/unified)
        py_grid: Reference to the parent Python grid object
    """
    def __init__(self,
                 neon_gate: neon.Gate,
                 grid_handle: ctypes.c_void_p,
                 cardinality: ctypes.c_int,
                 memory_type: neon.MemoryType,
                 dtype,
                 py_grid,
                 ):
        """
        Initialize a new multi-resolution field.
        
        Args:
            neon_gate (neon.Gate): Interface to the Neon C++ library
            grid_handle (ctypes.c_void_p): Handle to the parent grid C++ object
            cardinality (ctypes.c_int): Number of components per field element
            memory_type (neon.MemoryType): Memory allocation strategy (HOST, DEVICE, UNIFIED)
            dtype: Python data type for field elements (e.g., float, int, double)
            py_grid: Reference to the parent Python grid object
            
        Raises:
            Exception: If grid_handle is invalid (null pointer)
            Exception: If field initialization fails in C++ backend
        """
        # Store field configuration
        self.dtype = dtype
        if grid_handle == 0:
            raise InvalidFieldHandleError('Grid handle is invalid')

        # Core field attributes
        self._neon_gate: neon.Gate = neon_gate
        self.handle_type = ctypes.c_void_p
        self._handle: ctypes.c_uint64 = ctypes.c_void_p(0)  # Will be set by C++ constructor
        self._grid_handle = grid_handle
        self._cardinality = ctypes.c_int(cardinality)
        self._memory_type = memory_type
        self._py_grid = py_grid
        
        # Initialize field with C++ backend
        self._set_field_type()    # Determine C++ type mappings
        self._help_load_api()     # Load C++ API functions
        self._help_field_new()    # Create C++ field object

    def __del__(self):
        """Destructor - cleanup C++ resources when Python object is garbage collected."""
        self.cleanup()

    def __enter__(self) -> 'mField':
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
            self.help_delete()
        
        self._cleaned = True

    def _set_field_type(self):
        """
        Configure type-specific attributes based on the field's data type.
        
        Sets up:
        - type_mapping: Dictionary containing C++ type information
        - suffix: String suffix for C++ function names (e.g., '_f32', '_i32')
        - Partition_type: Corresponding partition class for this field type
        """
        self.type_mapping = self._neon_gate.get_type_mapping(self.dtype)
        self.suffix = f'_{self.type_mapping["suffix"]}'
        self.Partition_type = getattr(neon.multires.mPartition, f'mPartition{self.suffix}')

    def _help_load_api(self):
        """
        Load and configure C++ API function pointers from the shared library.
        
        This method sets up all the ctypes function signatures for:
        - Field creation and deletion
        - Data access operations (read/write)
        - Memory management (host/device transfers)
        - Data export functionality
        - Fill and copy operations
        """
        # Get reference to the shared library
        lib_obj = self._neon_gate.lib

        # === Field Lifecycle Management ===
        # Field creation API
        self.api_new = getattr(lib_obj, f'mGrid_mField_new{self.suffix}')
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),  # Output: field handle
                                 self.handle_type,                  # Input: grid handle
                                 ctypes.c_int,                      # Input: cardinality
                                 neon.MemoryType]                   # Input: memory type
        self.api_new.restype = ctypes.c_int

        # Field deletion API
        self.api_delete = getattr(lib_obj, f'mGrid_mField_delete{self.suffix}')
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]  # Input: field handle pointer
        self.api_delete.restype = ctypes.c_int

        # === Partition Management ===
        # Get field partition for specific execution context
        self.api_get_partition = getattr(lib_obj, f'mGrid_mField_get_partition{self.suffix}')
        self.api_get_partition.argtypes = [
            self.handle_type,                       # Input: field handle
            ctypes.POINTER(self.Partition_type),    # Output: partition object
            ctypes.c_int,                          # Input: resolution level
            neon.Execution,                        # Input: execution type (HOST/DEVICE)
            ctypes.c_int,                          # Input: device ID
            neon.DataView,                         # Input: data view (STANDARD/BOUNDARY)
        ]
        self.api_get_partition.restype = ctypes.c_int

        # # size partition
        # self.neon.lib.mGrid_mField_partition_size.argtypes = [
        #     ctypes.POINTER(self.Partition_type)]
        # self.neon.lib.mGrid_mField_partition_size.restype = ctypes.c_int

        # === Data Access Operations ===
        # Read field value at specific location
        self.api_read = getattr(lib_obj, f'mGrid_mField_read{self.suffix}')
        self.api_read.argtypes = [self.handle_type,                    # Input: field handle
                                  ctypes.c_int32,                      # Input: resolution level
                                  ctypes.POINTER(neon.Index_3d),       # Input: 3D index position
                                  ctypes.c_int32]                      # Input: cardinality component
        self.api_read.restype = self.type_mapping["ctype"]

        # Write field value at specific location
        self.api_write = getattr(lib_obj, f'mGrid_mField_write{self.suffix}')
        self.api_write.argtypes = [self.handle_type,                   # Input: field handle
                                   ctypes.c_int32,                     # Input: resolution level
                                   ctypes.POINTER(neon.Index_3d),      # Input: 3D index position
                                   ctypes.c_int32,                     # Input: cardinality component
                                   self.type_mapping["ctype"]]         # Input: value to write
        self.api_write.restype = ctypes.c_int

        # === Memory Management ===
        # Transfer data from device to host memory
        self.api_update_host = getattr(lib_obj, f'mGrid_mField_update_host_data{self.suffix}')
        self.api_update_host.argtypes = [self.handle_type,             # Input: field handle
                                         ctypes.c_int32]               # Input: stream ID
        self.api_update_host.restype = ctypes.c_int32

        # Transfer data from host to device memory
        self.api_update_device = getattr(lib_obj, f'mGrid_mField_update_device_data{self.suffix}')
        self.api_update_device.argtypes = [self.handle_type,           # Input: field handle
                                           ctypes.c_int32]             # Input: stream ID
        self.api_update_device.restype = ctypes.c_int32

        # === Data Export ===
        # Export field data to VTI format for visualization
        self.api_export_vti = getattr(lib_obj, f'mGrid_mField_to_vti{self.suffix}')
        self.api_export_vti.argtypes = [self.handle_type,              # Input: field handle
                                        ctypes.c_char_p,               # Input: filename
                                        ctypes.c_char_p,               # Input: field name
                                        ctypes.c_bool,                 # Input: output levels flag
                                        ctypes.c_bool,                 # Input: output block ID flag
                                        ctypes.c_bool]                 # Input: output voxel ID flag
        self.api_export_vti.restype = ctypes.c_int32

        # Export field data to VTI format (debug version)
        self.api_export_vti_debug = getattr(lib_obj, f'mGrid_mField_to_vti_debug{self.suffix}')
        self.api_export_vti_debug.argtypes = [self.handle_type,        # Input: field handle
                                              ctypes.c_char_p,         # Input: filename
                                              ctypes.c_char_p]         # Input: field name
        self.api_export_vti_debug.restype = ctypes.c_int32

        # === Field Operations ===
        # Fill field with a constant value
        self.api_fill = getattr(lib_obj, f'mGrid_mField_fill{self.suffix}')
        self.api_fill.argtypes = [self.handle_type,                    # Input: field handle
                                  ctypes.c_int32,                      # Input: resolution level
                                  self.type_mapping["ctype"],          # Input: fill value
                                  ctypes.c_int]                        # Input: stream ID
        self.api_fill.restype = ctypes.c_int

        # Copy data between fields
        self.api_copy = getattr(lib_obj, f'mGrid_mField_copy{self.suffix}')
        self.api_copy.argtypes = [self.handle_type,                    # Input: destination field handle
                                  self.handle_type,                    # Input: source field handle
                                  ctypes.c_int,                        # Input: resolution level
                                  ctypes.c_int]                        # Input: stream ID
        self.api_copy.restype = ctypes.c_int

    def _help_field_new(self):
        """
        Create the C++ field object and initialize the handle.
        
        Raises:
            Exception: If the field handle is invalid
            Exception: If C++ field creation fails
        """
        if self._handle == 0:
            raise InvalidFieldHandleError('Field handle is invalid')

        res = self.api_new(ctypes.pointer(self._handle),
                           self._grid_handle,
                           self._cardinality,
                           self._memory_type)
        if res != 0:
            raise FieldInitializationError(f'Failed to initialize field (error code: {res})')
        
        # Update public handle for backward compatibility
        self.handle = self._handle

    def help_delete(self):
        """
        Clean up C++ field resources.
        
        This method is called automatically by the destructor but can also
        be called manually to free resources earlier.
        
        Raises:
            Exception: If field deletion fails in C++ backend
        """
        if self._handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self._handle))
        if res != 0:
            raise FieldError(f'Failed to delete field (error code: {res})')

    def get_grid(self):
        """
        Get the parent grid object.
        
        Returns:
            mGrid: The parent grid object that owns this field
        """
        return self._py_grid

    def get_shape(self):
        """
        Get the 3D dimensions of the field.
        
        Returns:
            tuple: (x, y, z) dimensions as a tuple of integers
        """
        dim = self.get_grid().get_dimensions()
        return (dim.x, dim.y, dim.z)

    def get_partition(self,
                      level: ctypes.c_int,
                      execution: neon.Execution,
                      device_id: ctypes.c_int,
                      data_view: neon.DataView
                      ):
        """
        Get a field partition for parallel execution.
        
        A partition represents a subset of the field data that can be processed
        independently, enabling parallel computation across multiple threads/devices.
        
        Args:
            level (ctypes.c_int): Resolution level (0 = finest, higher = coarser)
            execution (neon.Execution): Execution context (HOST or DEVICE)
            device_id (ctypes.c_int): Device identifier for GPU execution
            data_view (neon.DataView): Data access pattern (STANDARD or BOUNDARY)
            
        Returns:
            Partition object: Type-specific partition for parallel processing
            
        Raises:
            Exception: If field handle is invalid
            Exception: If partition creation fails
        """
        if self._handle == 0:
            raise InvalidFieldHandleError('Field handle is invalid')

        partition = self.Partition_type()

        res = self.api_get_partition(self._handle,
                                     partition,
                                     level,
                                     execution,
                                     device_id,
                                     data_view)
        if res != 0:
            raise PartitionError(f'Failed to get partition (error code: {res})')

        # ccp_size = self.neon.lib.bGrid_bField_partition_size(partition)
        # ctypes_size = ctypes.sizeof(partition)
        #
        # if ccp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')
        #
        # # print(f"Partition {partition}")
        return partition

    def read(self,
             level: ctypes.c_int,
             idx: neon.Index_3d,
             cardinality: ctypes.c_int):
        """
        Read a value from the field at a specific location.
        
        Args:
            level (ctypes.c_int): Resolution level to read from
            idx (neon.Index_3d): 3D coordinate position
            cardinality (ctypes.c_int): Component index for multi-component fields
            
        Returns:
            Field value at the specified location (type depends on field dtype)
        """
        return self.api_read(self._handle,
                             level,
                             idx,
                             cardinality)

    def write(self,
              level: ctypes.c_int,
              idx: neon.Index_3d,
              cardinality: ctypes.c_int,
              newValue):
        """
        Write a value to the field at a specific location.
        
        Args:
            level (ctypes.c_int): Resolution level to write to
            idx (neon.Index_3d): 3D coordinate position
            cardinality (ctypes.c_int): Component index for multi-component fields
            newValue: Value to write (automatically converted to field's dtype)
            
        Returns:
            int: Status code (0 = success)
        """
        return self.api_write(self._handle,
                              level,
                              idx,
                              cardinality,
                              self.type_mapping['ctype'](newValue))

    def update_host(self,
                    stream: ctypes.c_int):
        """
        Transfer field data from device to host memory.
        
        This method synchronizes data that may have been modified on GPU
        back to CPU-accessible memory.
        
        Args:
            stream (ctypes.c_int): CUDA stream ID for asynchronous transfer
            
        Returns:
            int: Status code (0 = success)
        """
        return self.api_update_host(self._handle,
                                    stream)

    def update_device(self,
                      stream: ctypes.c_int):
        """
        Transfer field data from host to device memory.
        
        This method uploads data from CPU memory to GPU for computation.
        
        Args:
            stream (ctypes.c_int): CUDA stream ID for asynchronous transfer
            
        Returns:
            int: Status code (0 = success)
        """
        return self.api_update_device(self._handle,
                                      stream)

    def export_vti(self, filename: str,
                   field_name: str = "field",
                   outputLevels: bool = True,
                   outputBlockID: bool = True,
                   outputVoxelID: bool = True,
                   filterOverlaps: bool = True):
        """
        Export field data to VTI format for visualization.
        
        VTI (VTK Image Data) is a standard format for structured grid data
        that can be opened in ParaView, VisIt, and other visualization tools.
        
        Args:
            filename (str): Output filename (should end with .vti)
            field_name (str): Name for the field in the VTI file
            outputLevels (bool): Include multi-resolution level information
            outputBlockID (bool): Include block identifier data
            outputVoxelID (bool): Include voxel identifier data
            filterOverlaps (bool): Remove overlapping regions between levels
        """
        self.api_export_vti(self._handle, filename.encode('utf-8'), field_name.encode('utf-8'),
                            outputLevels,
                            outputBlockID,
                            outputVoxelID,
                            filterOverlaps)

    @property
    def cardinality(self) -> int:
        """Number of components per field element."""
        return self._cardinality.value

    @property
    def field_type(self) -> type:
        """Python data type of field elements."""
        return self.dtype

    @property
    def handle(self) -> ctypes.c_void_p:
        """C++ object handle (read-only)."""
        return self._handle

    @handle.setter
    def handle(self, value: ctypes.c_void_p) -> None:
        """Set the handle (for backward compatibility only)."""
        self._handle = value

    @property
    def grid(self):
        """Parent grid object."""
        return self._py_grid

    @property
    def shape(self) -> tuple:
        """3D dimensions of the field."""
        dim = self.grid.get_dimensions()
        return (dim.x, dim.y, dim.z)

    @property
    def memory_type(self) -> neon.MemoryType:
        """Memory allocation type."""
        return self._memory_type

    @property
    def name(self) -> str:
        """Field type name identifier."""
        return "mField"

    def copy_from_run(self, level, src_field, stream_idx):
        """
        Copy data from another field at runtime.
        
        This method performs an asynchronous copy operation that can be
        executed in parallel with other operations on different streams.
        
        Args:
            level (int): Resolution level to copy
            src_field (mField): Source field to copy data from
            stream_idx (int): CUDA stream index for asynchronous execution
        """
        self.api_copy(self._handle, src_field._handle, level, stream_idx)

    def fill_run(self, level, value, stream_idx):
        """
        Fill field with a constant value at runtime.
        
        This method performs an asynchronous fill operation that can be
        executed in parallel with other operations on different streams.
        
        Args:
            level (int): Resolution level to fill
            value: Constant value to fill with (converted to field's dtype)
            stream_idx (int): CUDA stream index for asynchronous execution
        """
        value = self.type_mapping['ctype'](value)
        self.api_fill(self.handle,
                      level,
                      value.value,
                      stream_idx
                      )

    def zero_run(self, level, stream_idx):
        """
        Fill field with zeros at runtime.
        
        Convenience method that fills the field with the zero value
        appropriate for the field's data type.
        
        Args:
            level (int): Resolution level to zero out
            stream_idx (int): CUDA stream index for asynchronous execution
        """
        # Debug output for type checking (commented out for performance)
        # print(f"zero_run: stream_idx type: {type(stream_idx)}, expected ctype: {ctypes.c_int}")
        self.fill_run(value=self.dtype(0), level=level, stream_idx=stream_idx)

    @property
    def type(self):
        """
        Property access to the field's data type.
        
        Returns:
            type: Python type object for field elements
        """
        return self.dtype

    # Enhanced Debugging and Introspection Methods
    def __repr__(self) -> str:
        """
        Detailed string representation for debugging.
        
        Returns:
            str: Comprehensive representation showing key field properties
        """
        try:
            grid_name = getattr(self._py_grid, 'name', 'Unknown')
            memory_type_str = getattr(self._memory_type, 'name', str(self._memory_type))
        except:
            grid_name = 'Unknown'
            memory_type_str = 'Unknown'
        
        return (f"mField(type={self.dtype.__name__}, "
                f"cardinality={self.cardinality}, "
                f"shape={self.shape}, "
                f"memory={memory_type_str}, "
                f"grid={grid_name}, "
                f"handle={hex(self._handle.value) if self._handle else 'None'})")

    def __str__(self) -> str:
        """
        User-friendly string representation.
        
        Returns:
            str: Human-readable description of the field
        """
        return (f"Multi-resolution Field: {self.dtype.__name__} "
                f"{'scalar' if self.cardinality == 1 else f'vector({self.cardinality})'}, "
                f"shape {self.shape}")

    def get_field_info(self) -> dict:
        """
        Get comprehensive field information.
        
        Returns:
            dict: Dictionary containing field properties and statistics
        """
        try:
            grid_info = {
                'type': type(self._py_grid).__name__,
                'dimensions': self.shape,
                'num_levels': getattr(self._py_grid, 'num_levels', 'Unknown')
            }
        except:
            grid_info = {'type': 'Unknown', 'dimensions': 'Unknown', 'num_levels': 'Unknown'}

        info = {
            'field_type': self.name,
            'data_type': self.dtype.__name__,
            'cardinality': self.cardinality,
            'shape': self.shape,
            'memory_type': str(self._memory_type),
            'handle_value': hex(self._handle.value) if self._handle else 'None',
            'is_scalar': self.cardinality == 1,
            'is_vector': self.cardinality > 1,
            'grid_info': grid_info,
            'type_mapping': getattr(self, 'type_mapping', {}),
            'is_cleaned': getattr(self, '_cleaned', False)
        }
        
        return info

    def get_memory_info(self) -> dict:
        """
        Get memory-related information for the field.
        
        Returns:
            dict: Dictionary containing memory usage estimates
            
        Note:
            This provides estimates based on field structure.
            Actual C++ memory usage may differ.
        """
        try:
            element_size = getattr(self.dtype(), 'nbytes', 8)  # Default to 8 bytes if unknown
        except:
            element_size = 8

        shape = self.shape
        if isinstance(shape, tuple) and len(shape) == 3:
            total_elements = shape[0] * shape[1] * shape[2]
        else:
            total_elements = 0

        estimated_bytes = total_elements * element_size * self.cardinality

        return {
            'data_type': self.dtype.__name__,
            'element_size_bytes': element_size,
            'cardinality': self.cardinality,
            'shape': shape,
            'total_elements': total_elements,
            'estimated_field_bytes': estimated_bytes,
            'memory_type': str(self._memory_type),
            'estimated_total_mb': round(estimated_bytes / (1024 * 1024), 2)
        }

    def validate_field_integrity(self) -> bool:
        """
        Validate the integrity of the field structure.
        
        Returns:
            bool: True if field structure is valid, False otherwise
            
        Raises:
            FieldError: If critical integrity issues are found
        """
        try:
            # Check basic structure
            if not hasattr(self, 'dtype') or self.dtype is None:
                raise FieldError("Field must have a valid data type")
            
            if self.cardinality <= 0:
                raise FieldError("Field cardinality must be positive")
            
            # Check handle validity
            if not self._handle or self._handle.value == 0:
                raise InvalidFieldHandleError("Invalid C++ field handle")
            
            # Check grid reference
            if not self._py_grid:
                raise FieldError("Field must have a valid grid reference")
            
            # Check memory type
            if not self._memory_type:
                raise FieldError("Field must have a valid memory type")
            
            # Check type mapping
            if not hasattr(self, 'type_mapping') or not self.type_mapping:
                raise FieldError("Field must have valid type mapping")
            
            return True
            
        except FieldError:
            raise
        except Exception as e:
            raise FieldError(f"Field integrity validation failed: {e}")

    def get_debug_info(self) -> dict:
        """
        Get comprehensive debugging information.
        
        Returns:
            dict: Dictionary containing all available debug information
        """
        debug_info = {
            'field_info': self.get_field_info(),
            'memory_info': self.get_memory_info(),
            'type_mapping': getattr(self, 'type_mapping', {}),
            'suffix': getattr(self, 'suffix', 'Unknown'),
            'partition_type': str(getattr(self, 'Partition_type', 'Unknown')),
            'neon_gate_info': {
                'type': type(self._neon_gate).__name__ if self._neon_gate else 'None',
                'lib_available': hasattr(self._neon_gate, 'lib') if self._neon_gate else False
            }
        }
        
        return debug_info
