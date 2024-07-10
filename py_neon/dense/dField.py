import ctypes

import py_neon
from py_neon.dataview import DataView as NeDataView
from py_neon.execution import Execution as NeExecution
from py_neon.index_3d import Index_3d
from py_neon.py_ne import Py_neon as NePy_neon
from .dPartition import dPartitionInt as NeDPartition


class dField(object):
    def __init__(self,
                 py_neon: NePy_neon,
                 grid_handle: ctypes.c_uint64,
                 py_grid,
                 ):

        if grid_handle == 0:
            raise Exception('DField: Invalid handle')

        self.py_neon = py_neon
        self.handle_type = ctypes.POINTER(ctypes.c_uint64)
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.grid_handle = grid_handle
        self.help_load_api()
        self.help_new()
        self.py_grid = py_grid

    def __del__(self):
        self.help_delete()

    def help_load_api(self):
        # Importing new functions
        ## new_field
        self.py_neon.lib.dGrid_dField_new.argtypes = [self.handle_type,
                                                      self.handle_type]
        self.py_neon.lib.dGrid_dField_new.restype = ctypes.c_int

        ## delete_field
        self.py_neon.lib.dGrid_dField_delete.argtypes = [self.handle_type]
        self.py_neon.lib.dGrid_dField_delete.restype = ctypes.c_int

        ## get_partition
        self.py_neon.lib.dGrid_dField_get_partition.argtypes = [self.handle_type,
                                                                ctypes.POINTER(NeDPartition),  # the span object
                                                                NeExecution,  # the execution type
                                                                ctypes.c_int,  # the device id
                                                                NeDataView,  # the data view
                                                                ]
        self.py_neon.lib.dGrid_dField_get_partition.restype = ctypes.c_int

        # size partition
        self.py_neon.lib.dGrid_dField_partition_size.argtypes = [ctypes.POINTER(NeDPartition)]
        self.py_neon.lib.dGrid_dField_partition_size.restype = ctypes.c_int

        # field read
        self.py_neon.lib.dGrid_dField_read.argtypes = [self.handle_type,
                                                       py_neon.Index_3d,
                                                       ctypes.c_int]
        self.py_neon.lib.dGrid_dField_read.restype = ctypes.c_int

        # field write
        self.py_neon.lib.dGrid_dField_write.argtypes = [self.handle_type,
                                                        py_neon.Index_3d,
                                                        ctypes.c_int,
                                                        ctypes.c_int]
        self.py_neon.lib.dGrid_dField_write.restype = ctypes.c_int

        # field update host data
        self.py_neon.lib.dGrid_dField_update_host_data.argtypes = [self.handle_type,
                                                                   ctypes.c_int]
        self.py_neon.lib.dGrid_dField_update_host_data.restype = ctypes.c_int

        # field update device data
        self.py_neon.lib.dGrid_dField_update_device_data.argtypes = [self.handle_type,
                                                                     ctypes.c_int]
        self.py_neon.lib.dGrid_dField_update_device_data.restype = ctypes.c_int

    def help_new(self):
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        res = self.py_neon.lib.dGrid_dField_new(self.handle, self.grid_handle)
        if res != 0:
            raise Exception('DGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dGrid_dField_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete field')

    def get_grid(self):
        return self.py_grid

    def get_partition(self,
                      execution: NeExecution,
                      c: ctypes.c_int,
                      data_view: NeDataView
                      ) -> NeDPartition:

        if self.handle == 0:
            raise Exception('DField: Invalid handle')

        partition = NeDPartition()

        res = self.py_neon.lib.dGrid_dField_get_partition(self.handle,
                                                          partition,
                                                          execution,
                                                          c,
                                                          data_view)
        if res != 0:
            raise Exception('Failed to get span')

        ccp_size = self.py_neon.lib.dGrid_dField_partition_size(partition)
        ctypes_size = ctypes.sizeof(partition)

        if ccp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')

        print(f"Partition {partition}")
        return partition

    def read(self, idx: Index_3d, cardinality: int):
        return self.py_neon.lib.dGrid_dField_read(self.handle, idx, cardinality)

    def write(self, idx: Index_3d, cardinality: int, newValue: int):
        return self.py_neon.lib.dGrid_dField_write(self.handle, idx, cardinality, newValue)

    def updateHostData(self, streamSetId: int):
        return self.py_neon.lib.dGrid_dField_update_host_data(self.handle, streamSetId)

    def updateDeviceData(self, streamSetId: int):
        return self.py_neon.lib.dGrid_dField_update_device_data(self.handle, streamSetId)
