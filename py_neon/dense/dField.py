import ctypes

import py_neon
import py_neon.dense.dPartition as dPartition
from py_neon.dataview import DataView as NeDataView
from py_neon.execution import Execution as NeExecution
from py_neon.index_3d import Index_3d
from py_neon.py_ne import Py_neon as NePy_neon


# from .dPartition import dPartitionInt as dPartitionInt


class dField(object):
    def __init__(self,
                 py_neon: NePy_neon,
                 grid_handle: ctypes.c_void_p,
                 cardinality: ctypes.c_int,
                 dtype,
                 py_grid,
                 ):

        self.dtype = dtype
        if grid_handle == 0:
            raise Exception('DField: Invalid handle')

        self.py_neon = py_neon
        self.handle_type = ctypes.c_void_p
        self.handle: ctypes.c_uint64 = ctypes.c_void_p(0)
        self.grid_handle = grid_handle
        self.cardinality = cardinality
        self.py_grid = py_grid
        self.field_type = None
        self._set_field_type()
        self._help_load_api()
        self._help_field_new()

    def __del__(self):
        self.help_delete()
        pass

    def _set_field_type(self):
        if self.dtype == int:
            self.field_type = ctypes.c_int32
            self.Partition_type = dPartition.dPartitionInt
            self.suffix = '_int'
        elif self.dtype == float:
            self.field_type = ctypes.c_float
            self.Partition_type = dPartition.dPartitionFloat
            self.suffix = '_float'
        elif self.dtype == bool:
            self.field_type = ctypes.c_char
            self.Partition_type = dPartition.dPartitionChar
        elif self.dtype == ctypes.c_double:
            self.field_type = ctypes.c_double
            self.Partition_type = dPartition.dPartitionDouble
            self.suffix = '_double'
        elif self.dtype == ctypes.c_float:
            self.field_type = ctypes.c_float
            self.Partition_type = dPartition.dPartitionFloat
            self.suffix = '_float'
        else:
            raise Exception('dField: Unsupported data type')

    def _help_load_api(self):
        # Importing new functions
        ## new_field
        lib_obj = self.py_neon.lib

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
            NeExecution,  # the execution type
            ctypes.c_int,  # the device id
            NeDataView,  # the data view
        ]
        self.api_get_partition.restype = ctypes.c_int

        # size partition
        self.py_neon.lib.dGrid_dField_partition_size.argtypes = [
            ctypes.POINTER(self.Partition_type)]
        self.py_neon.lib.dGrid_dField_partition_size.restype = ctypes.c_int

        # field read
        self.api_read = getattr(lib_obj, f'dGrid_dField_read{self.suffix}')
        self.api_read.argtypes = [self.handle_type,
                                  ctypes.POINTER(py_neon.Index_3d),
                                  ctypes.c_int]
        self.api_read.restype = self.field_type

        # field write
        self.api_write = getattr(lib_obj, f'dGrid_dField_write{self.suffix}')
        self.api_write.argtypes = [self.handle_type,
                                   ctypes.POINTER(py_neon.Index_3d),
                                   ctypes.c_int,
                                   self.field_type]
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

    def _help_field_new(self):
        if self.handle == 0:
            raise Exception('dGrid: Invalid handle')

        res = self.api_new(ctypes.pointer(self.handle),
                           self.grid_handle,
                           self.cardinality)
        if res != 0:
            raise Exception('dGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete field')

    def get_grid(self):
        return self.py_grid

    def get_partition(self,
                      execution: NeExecution,
                      c: ctypes.c_int,
                      data_view: NeDataView
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

        ccp_size = self.py_neon.lib.dGrid_dField_partition_size(partition)
        ctypes_size = ctypes.sizeof(partition)

        if ccp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')

        # print(f"Partition {partition}")
        return partition

    def read(self, idx: Index_3d, cardinality: ctypes.c_int):
        return self.api_read(self.handle,
                             idx,
                             cardinality)

    def write(self, idx: Index_3d, cardinality: ctypes.c_int, newValue):
        return self.api_write(self.handle,
                              idx,
                              cardinality,
                              self.field_type(newValue))

    def updateHostData(self, streamSetId: ctypes.c_int):
        return self.api_update_host(self.handle,
                                    streamSetId)

    def updateDeviceData(self, streamSetId: ctypes.c_int):
        return self.api_update_device(self.handle, streamSetId)
