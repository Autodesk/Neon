"""
Dense Field Implementation for Neon Computing Framework

This module provides the dField class, which represents a multi-dimensional
data field for dense grid structures. The field supports various data types
and provides operations for data manipulation and parallel computing.
"""

import ctypes
from typing import Optional, Union
from enum import Enum

import neon
import warp as wp

# from .dPartition import dPartitionInt as dPartitionInt


class FieldError(Exception):
    """Base exception for field operations."""
    pass


class InvalidFieldHandleError(FieldError):
    """Raised when field handle operations fail."""
    pass


class FieldInitializationError(FieldError):
    """Raised when field initialization fails."""
    pass


class DataTransferError(FieldError):
    """Raised when data transfer operations fail."""
    pass


class dField(object):
    def __init__(self,
                 neon_gate: neon.Gate,
                 grid_handle: ctypes.c_void_p,
                 cardinality: ctypes.c_int,
                 dtype,
                 py_grid,
                 ):

        # Store field configuration
        self.dtype = dtype
        if grid_handle == 0:
            raise InvalidFieldHandleError('Grid handle is invalid')

        # Core field attributes
        self._neon_gate: neon.Gate = neon_gate
        self.handle_type = ctypes.c_void_p
        self._handle: ctypes.c_uint64 = ctypes.c_void_p(0)  # Will be set by C++ constructor
        self._grid_handle = grid_handle
        self._cardinality = cardinality
        self._py_grid = py_grid
        
        # Initialize field with C++ backend
        self._set_field_type()    # Determine C++ type mappings
        self._help_load_api()     # Load C++ API functions
        self._help_field_new()    # Create C++ field object

    def __del__(self):
        """Destructor - cleanup C++ resources when Python object is garbage collected."""
        self.cleanup()

    def __enter__(self) -> 'dField':
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
        self.Partition_type = getattr(neon.dense.dPartition, f'dPartition{self.suffix}')

    def _help_load_api(self):
        """
        Load and configure C++ API function bindings for this field type.
        
        Sets up ctypes bindings for all field operations including:
        - Field lifecycle management (new/delete)
        - Partition management
        - Data access operations
        - Memory synchronization
        - Fill and copy operations
        """
        # Get reference to the shared library
        lib_obj = self._neon_gate.lib

        # ---------------------------------------------------------------------
        self.api_new = getattr(lib_obj, f'dGrid_dField_new{self.suffix}')
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),
                                 self.handle_type,
                                 ctypes.c_int]
        self.api_new.restype = ctypes.c_int

        # ---------------------------------------------------------------------
        self.api_delete = getattr(lib_obj, f'dGrid_dField_delete{self.suffix}')
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.api_delete.restype = ctypes.c_int

        ## get_partition
        self.api_get_partition = getattr(lib_obj, f'dGrid_dField_get_partition{self.suffix}')
        self.api_get_partition.argtypes = [
            self.handle_type,
            ctypes.POINTER(self.Partition_type),  # the span object
            neon.Execution,  # the execution type
            ctypes.c_int,  # the device id
            neon.DataView,  # the data view
        ]
        self.api_get_partition.restype = ctypes.c_int

        # # size partition
        # self.neon.lib.dGrid_dField_partition_size.argtypes = [
        #     ctypes.POINTER(self.Partition_type)]
        # self.neon.lib.dGrid_dField_partition_size.restype = ctypes.c_int

        # field read
        self.api_read = getattr(lib_obj, f'dGrid_dField_read{self.suffix}')
        self.api_read.argtypes = [self.handle_type,
                                  ctypes.POINTER(neon.Index_3d),
                                  ctypes.c_int]
        self.api_read.restype = self.type_mapping["ctype"]

        # field write
        self.api_write = getattr(lib_obj, f'dGrid_dField_write{self.suffix}')
        self.api_write.argtypes = [self.handle_type,
                                   ctypes.POINTER(neon.Index_3d),
                                   ctypes.c_int,
                                   self.type_mapping["ctype"]]
        self.api_write.restype = ctypes.c_int

        # field update host data
        self.api_update_host = getattr(lib_obj, f'dGrid_dField_update_host_data{self.suffix}')
        self.api_update_host.argtypes = [self.handle_type,
                                         ctypes.c_int]
        self.api_update_host.restype = ctypes.c_int

        # field update device data
        self.api_update_device = getattr(lib_obj, f'dGrid_dField_update_device_data{self.suffix}')
        self.api_update_device.argtypes = [self.handle_type,
                                           ctypes.c_int]
        self.api_update_device.restype = ctypes.c_int

        # export vti
        self.api_export_vti = getattr(lib_obj, f'dGrid_dField_to_vti{self.suffix}')
        self.api_export_vti.argtypes = [self.handle_type,
                                           ctypes.c_char_p,
                                           ctypes.c_char_p]
        self.api_export_vti.restype = ctypes.c_int

        # field update host data
        self.api_fill = getattr(lib_obj, f'dGrid_dField_fill{self.suffix}')
        self.api_fill.argtypes = [self.handle_type,
                                   self.type_mapping["ctype"],
                                  ctypes.c_int]
        self.api_fill.restype = ctypes.c_int

        # self.api_fill = getattr(lib_obj, f'dGrid_dField_fill{self.suffix}')
        # self.api_fill.argtypes = [self.handle_type,
        #                           self.type_mapping["ctype"],
        #                           ctypes.c_int]
        # self.api_fill.restype = ctypes.c_int

        # field update host data
        self.api_copy = getattr(lib_obj, f'dGrid_dField_copy{self.suffix}')
        self.api_copy.argtypes = [self.handle_type,
                                         self.handle_type,
                                         ctypes.c_int]
        self.api_copy.restype = ctypes.c_int

    def _help_field_new(self):
        """
        Create and initialize the underlying C++ field object.
        
        Raises:
            FieldInitializationError: If field handle is invalid or creation fails.
        """
        if self._handle == 0:
            raise FieldInitializationError('Invalid field handle')

        res = self.api_new(ctypes.pointer(self._handle),
                           self._grid_handle,
                           self._cardinality)
        if res != 0:
            raise FieldInitializationError('Failed to initialize field')

    def help_delete(self):
        """
        Clean up and destroy the underlying C++ field object.
        
        Raises:
            FieldError: If field deletion fails.
        """
        if self._handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self._handle))
        if res != 0:
            raise FieldError('Failed to delete field')

    def get_grid(self):
        """Get the parent grid object."""
        return self._py_grid

    def get_shape(self):
        dim =  self.get_grid().get_dimensions()
        return (dim.x, dim.y, dim.z)

    def get_partition(self,
                      execution: neon.Execution,
                      c: ctypes.c_int,
                      data_view: neon.DataView
                      ):
        if self.handle == 0:
            raise Exception('dField: Invalid handle')

        partition = self.Partition_type()

        res = self.api_get_partition(self.handle,
                                     partition,
                                     execution,
                                     c,
                                     data_view)
        if res != 0:
            raise Exception('Failed to get partition')

        # ccp_size = self.neon.lib.dGrid_dField_partition_size(partition)
        # ctypes_size = ctypes.sizeof(partition)
        #
        # if ccp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')
        #
        # # print(f"Partition {partition}")
        return partition

    def get_partition_type(self):
        return self.Partition_type

    def read(self, idx: neon.Index_3d, cardinality: ctypes.c_int):
        return self.api_read(self.handle,
                             idx,
                             cardinality)

    def write(self, idx: neon.Index_3d, cardinality: ctypes.c_int, newValue):
        return self.api_write(self.handle,
                              idx,
                              cardinality,
                              self.type_mapping['ctype'](newValue))

    def update_host(self, streamSetId: ctypes.c_int):
        return self.api_update_host(self.handle,
                                    streamSetId)

    def update_device(self, streamSetId: ctypes.c_int):
        return self.api_update_device(self.handle,
                                      streamSetId)

    def export_vti(self, filename: str,
                   field_name: str = "field"):
        self.api_export_vti(self.handle, filename.encode('utf-8'), field_name.encode('utf-8'))

    def get_cardinality(self):
        return self.cardinality.value

    def get_type(self):
        return self.dtype

    def get_handle(self):
        return self.handle

    def copy_from_run(self, src_field, stream_idx):
        self.api_copy(self.handle, src_field.handle, stream_idx)

    def fill_run(self, value, stream_idx):
        value = self.type_mapping['ctype'](value)
        # print(f"fill_run: value type: {type(value)}, expected ctype: {self.type_mapping['ctype']}")
        # print(f"fill_run: stream_idx type: {type(stream_idx)}, expected ctype: {ctypes.c_int}")

        self.api_fill(self.get_handle(),
                      value.value,
                      stream_idx
                      )

    def zero_run(self, stream_idx):
        # print(f"zero_run: stream_idx type: {type(stream_idx)}, expected ctype: {ctypes.c_int}")
        self.fill_run(value=self.dtype(0), stream_idx=stream_idx)

    @property
    def type(self):
        return self.dtype
