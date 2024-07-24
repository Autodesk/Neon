import copy
import ctypes
from enum import Enum
import py_neon
from py_neon import Py_neon
from .mPartition import mPartitionInt as NeMPartitionInt
from py_neon.execution import Execution as NeExecution
from py_neon.dataview import DataView as NeDataView
from py_neon.py_ne import Py_neon as NePy_neon
from py_neon.index_3d import Index_3d

class mField(object):
    def __init__(self,
                 grid_handle: ctypes.c_uint64,
                 cardinality: ctypes.c_int
                 ):

        if grid_handle == 0:
            raise Exception('DField: Invalid handle')
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.handle_type = ctypes.POINTER(ctypes.c_uint64)
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.grid_handle = grid_handle
        self.cardinality = cardinality
        self._help_load_api()
        self._help_field_new()

    def __del__(self):
        self.help_delete()

    def _help_load_api(self):
        # Importing new functions
        ## new_field
        self.py_neon.lib.mGrid_mField_new.argtypes = [self.handle_type,
                                                      self.handle_type]
        self.py_neon.lib.mGrid_mField_new.restype = ctypes.c_int

        ## delete_field
        self.py_neon.lib.mGrid_mField_delete.argtypes = [self.handle_type]
        self.py_neon.lib.mGrid_mField_delete.restype = ctypes.c_int

        ## get_partition
        self.py_neon.lib.mGrid_mField_get_partition.argtypes = [self.handle_type,
                                                                ctypes.POINTER(NeMPartitionInt),  # the span object
                                                                ctypes.c_int,
                                                                NeExecution,  # the execution type
                                                                ctypes.c_int,  # the device id
                                                                NeDataView,  # the data view
                                                                ]
        self.py_neon.lib.mGrid_mField_get_partition.restype = ctypes.c_int

        # size partition
        self.py_neon.lib.mGrid_mField_partition_size.argtypes = [ctypes.POINTER(NeMPartitionInt)]
        self.py_neon.lib.mGrid_mField_partition_size.restype = ctypes.c_int

        # field read
        self.py_neon.lib.mGrid_mField_read.argtypes = [self.handle_type,
                                                       ctypes.c_int,
                                                       ctypes.POINTER(py_neon.Index_3d),
                                                       ctypes.c_int]
        self.py_neon.lib.mGrid_mField_read.restype = ctypes.c_int

        # field write
        self.py_neon.lib.mGrid_mField_write.argtypes = [self.handle_type,
                                                        ctypes.c_int,
                                                        ctypes.POINTER(py_neon.Index_3d),
                                                        ctypes.c_int,
                                                        ctypes.c_int]
        self.py_neon.lib.mGrid_mField_write.restype = ctypes.c_int

        # field update host data
        self.py_neon.lib.mGrid_mField_update_host_data.argtypes = [self.handle_type,
                                                       ctypes.c_int]
        self.py_neon.lib.mGrid_mField_update_host_data.restype = ctypes.c_int

        # field update device data
        self.py_neon.lib.mGrid_mField_update_device_data.argtypes = [self.handle_type,
                                                       ctypes.c_int]
        self.py_neon.lib.mGrid_mField_update_device_data.restype = ctypes.c_int


    def _help_field_new(self):
        if self.handle == 0:
            raise Exception('mGrid: Invalid handle')

        res = self.py_neon.lib.mGrid_mField_new(ctypes.byref(self.handle), ctypes.byref(self.grid_handle), self.cardinality)
        if res != 0:
            raise Exception('mGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.mGrid_mField_delete(ctypes.byref(self.handle))
        if res != 0:
            raise Exception('Failed to delete field')

    def get_partition(self,
                      field_index: ctypes.c_int,
                      execution: NeExecution,
                      c: ctypes.c_int,
                      data_view: NeDataView
                      ) -> NeMPartitionInt:

        if self.handle == 0:
            raise Exception('mField: Invalid handle')

        partition = NeMPartitionInt()

        res = self.py_neon.lib.mGrid_mField_get_partition(self.handle,
                                                          partition,
                                                          field_index,
                                                          execution,
                                                          c,
                                                          data_view)
        if res != 0:
            raise Exception('Failed to get partition')

        ccp_size = self.py_neon.lib.mGrid_mField_partition_size(partition)
        ctypes_size = ctypes.sizeof(partition)

        if ccp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')

        print(f"Partition {partition}")
        return partition
    
    def read(self, field_level: ctypes.c_int, idx: Index_3d, cardinality: ctypes.c_int):
        return self.py_neon.lib.mGrid_mField_read(ctypes.byref(self.handle), field_level, idx, cardinality)
    
    def write(self, field_level: ctypes.c_int, idx: Index_3d, cardinality: ctypes.c_int, newValue: ctypes.c_int):
        return self.py_neon.lib.mGrid_mField_write(ctypes.byref(self.handle), field_level, idx, cardinality, newValue)

    def updateHostData(self, streamSetId: ctypes.c_int):
        return self.py_neon.lib.mGrid_mField_update_host_data(ctypes.byref(self.handle), streamSetId)
    
    def updateDeviceData(self, streamSetId: ctypes.c_int):
        return self.py_neon.lib.mGrid_mField_update_device_data(ctypes.byref(self.handle), streamSetId)