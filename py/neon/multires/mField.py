"""
Multi-resolution Field Implementation for Neon Computing Framework

This module provides the mField class, which represents a multi-resolution field
data structure for parallel computing applications. The field supports various
data types and memory configurations for both CPU and GPU execution.

"""

import ctypes
import neon
import neon.multires.mPartition


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
            raise Exception('mField: Invalid handle')

        # Core field attributes
        self.neon_gate: neon.Gate = neon_gate
        self.handle_type = ctypes.c_void_p
        self.handle: ctypes.c_uint64 = ctypes.c_void_p(0)  # Will be set by C++ constructor
        self.grid_handle = grid_handle
        self.cardinality = ctypes.c_int(cardinality)
        self.memory_type = memory_type
        self.py_grid = py_grid
        
        # Initialize field with C++ backend
        self._set_field_type()    # Determine C++ type mappings
        self._help_load_api()     # Load C++ API functions
        self._help_field_new()    # Create C++ field object

    def __del__(self):
        """Destructor - cleanup C++ resources when Python object is garbage collected."""
        self.help_delete()

    def _set_field_type(self):
        """
        Configure type-specific attributes based on the field's data type.
        
        Sets up:
        - type_mapping: Dictionary containing C++ type information
        - suffix: String suffix for C++ function names (e.g., '_f32', '_i32')
        - Partition_type: Corresponding partition class for this field type
        """
        self.type_mapping = self.neon_gate.get_type_mapping(self.dtype)
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
        lib_obj = self.neon_gate.lib

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
        if self.handle == 0:
            raise Exception('mField: Invalid handle')

        res = self.api_new(ctypes.pointer(self.handle),
                           self.grid_handle,
                           self.cardinality,
                           self.memory_type)
        if res != 0:
            raise Exception('mField: Failed to initialize field')

    def help_delete(self):
        """
        Clean up C++ field resources.
        
        This method is called automatically by the destructor but can also
        be called manually to free resources earlier.
        
        Raises:
            Exception: If field deletion fails in C++ backend
        """
        if self.handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete field')

    def get_grid(self):
        """
        Get the parent grid object.
        
        Returns:
            mGrid: The parent grid object that owns this field
        """
        return self.py_grid

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
        if self.handle == 0:
            raise Exception('bField: Invalid handle')

        partition = self.Partition_type()

        res = self.api_get_partition(self.handle,
                                     partition,
                                     level,
                                     execution,
                                     device_id,
                                     data_view)
        if res != 0:
            raise Exception('Failed to get partition')

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
        return self.api_read(self.handle,
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
        return self.api_write(self.handle,
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
        return self.api_update_host(self.handle,
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
        return self.api_update_device(self.handle,
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
        self.api_export_vti(self.handle, filename.encode('utf-8'), field_name.encode('utf-8'),
                            outputLevels,
                            outputBlockID,
                            outputVoxelID,
                            filterOverlaps)

    def get_cardinality(self):
        """
        Get the number of components per field element.
        
        Returns:
            int: Cardinality (1 = scalar, 3 = vector, etc.)
        """
        return self.cardinality.value

    def get_type(self):
        """
        Get the Python data type of field elements.
        
        Returns:
            type: Python type object (e.g., float, int, np.float32)
        """
        return self.dtype

    def get_handle(self):
        """
        Get the C++ object handle.
        
        Returns:
            ctypes.c_void_p: Handle to the underlying C++ field object
        """
        return self.handle

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
        self.api_copy(self.handle, src_field.handle, level, stream_idx)

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
        self.api_fill(self.get_handle(),
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
