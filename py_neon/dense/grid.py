import copy
import ctypes
from enum import Enum

import py_neon
from .field import Field
from py_neon.execution import Execution
from py_neon import Py_neon
from py_neon.dataview import DataView
from .span import Span


class Grid(object):
    def __init__(self):
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_grid_new()

    def __del__(self):
        if self.handle == 0:
            return
        self.help_grid_delete()

    def help_load_api(self):

        # grid_new
        self.py_neon.lib.dGrid_new.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dGrid_new.restype = ctypes.c_int

        # grid_delete
        self.py_neon.lib.dGrid_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dGrid_delete.restype = ctypes.c_int

        self.py_neon.lib.dGrid_get_span.argtypes = [self.py_neon.handle_type,
                                                    ctypes.POINTER(Span),  # the span object
                                                    py_neon.Execution,  # the execution type
                                                    ctypes.c_int,  # the device id
                                                    py_neon.DataView,  # the data view
                                                    ]
        self.py_neon.lib.dGrid_get_span.restype = ctypes.c_int

    def help_grid_new(self):
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        res = self.py_neon.lib.dGrid_new(self.handle)
        if res != 0:
            raise Exception('DGrid: Failed to initialize grid')

    def help_grid_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dGrid_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete grid')

    def new_field(self) -> Field:
        field = Field(self.py_neon, self.handle)
        return field

    def get_span(self,
                 execution: Execution,
                 c: ctypes.c_int,
                 data_view: DataView) -> Span:
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        span = Span()
        res = self.py_neon.lib.dGrid_get_span(self.handle, span, execution, c, data_view)
        if res != 0:
            raise Exception('Failed to get span')
        return span
