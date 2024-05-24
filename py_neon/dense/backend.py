import copy
import ctypes
from enum import Enum

import py_neon
from py_neon.execution import Execution
from py_neon import Py_neon
from py_neon.dataview import DataView


class Backend(object):
    def __init__(self):
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_backend_new()

    def __del__(self):
        if self.handle == 0:
            return
        self.help_backend_delete()

    def help_load_api(self):

        # backend_new
        self.py_neon.lib.dBackend_new_default.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_new_default.restype = ctypes.c_int

        # backend_delete
        self.py_neon.lib.dBackend_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_delete.restype = ctypes.c_int



    def help_backend_new(self):
        if self.handle == 0:
            raise Exception('DBackend: Invalid handle')

        res = self.py_neon.lib.dBackend_new_default(self.handle)
        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')

    def help_backend_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dBackend_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete backend')
