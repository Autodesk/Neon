import copy
import ctypes
from enum import Enum
from .partition import PartitionInt as NePartitionInt
from py_neon.execution import Execution as NeExecution
from py_neon.dataview import DataView as NeDataView
from py_neon.py_ne import Py_neon as NePy_neon
from wpne.dense.partition import NeonDensePartitionInt as Wpne_NeonDensePartitionInt


class Field(object):
    def __init__(self,
                 py_neon: NePy_neon,
                 grid_handle: ctypes.c_uint64
                 ):

        if grid_handle == 0:
            raise Exception('DField: Invalid handle')

        self.py_neon = py_neon
        self.handle_type = ctypes.POINTER(ctypes.c_uint64)
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.grid_handle = grid_handle
        self.help_load_api()
        self.help_new()

    def __del__(self):
        self.help_delete()

    def help_load_api(self):
        # Importing new functions
        ## new_field
        self.py_neon.lib.dGrid_dField_new.argtypes = [self.handle_type,
                                                      self.handle_type]
        ## delete_field
        self.py_neon.lib.dGrid_dField_delete.argtypes = [self.handle_type]

        ## get_partition
        self.py_neon.lib.dGrid_dField_get_partition.argtypes = [self.handle_type,
                                                                ctypes.POINTER(NePartitionInt),  # the span object
                                                                NeExecution,  # the execution type
                                                                ctypes.c_int,  # the device id
                                                                NeDataView,  # the data view
                                                                ]

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

    def get_partition(self,
                      execution: NeExecution,
                      c: ctypes.c_int,
                      data_view: NeDataView
                      ) -> Wpne_NeonDensePartitionInt:

        if self.handle == 0:
            raise Exception('DField: Invalid handle')

        partition = NePartitionInt()

        res = self.py_neon.lib.dGrid_dField_get_partition(self.handle,
                                                          partition,
                                                          execution,
                                                          c,
                                                          data_view)
        if res != 0:
            raise Exception('Failed to get span')

        wpne_partition = Wpne_NeonDensePartitionInt(partition)
        return wpne_partition
