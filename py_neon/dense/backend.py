import copy
import ctypes
from enum import Enum

import py_neon
from py_neon.execution import Execution
from py_neon import Py_neon
from py_neon.dataview import DataView


class Backend(object):

    class Runtime(Enum):
        none = 0
        system = 0
        stream = 1
        openmp = 2

    def __init__(self, arg1 = None, arg2 = None):
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_backend_new(arg1, arg2)

    def __del__(self):
        if self.handle == 0:
            return
        self.help_backend_delete()

    def help_load_api(self):

        # backend_new
        self.py_neon.lib.dBackend_new1.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_new1.restype = ctypes.c_int

        # backend_new
        self.py_neon.lib.dBackend_new2.argtypes = [self.py_neon.handle_type,
                                                  ctypes.c_int,
                                                  ctypes.c_int]
        self.py_neon.lib.dBackend_new2.restype = ctypes.c_int

        # backend_new
        self.py_neon.lib.dBackend_new3.argtypes = [self.py_neon.handle_type,
                                                     ctypes.POINTER(ctypes.c_int),
                                                     ctypes.c_int]
        self.py_neon.lib.dBackend_new3.restype = ctypes.c_int

        # backend_delete
        self.py_neon.lib.dBackend_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_delete.restype = ctypes.c_int

        # backend_get_string
        self.py_neon.lib.dBackend_get_string.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_get_string.restype = ctypes.c_char_p



    def help_backend_new(self, arg1=None, arg2=None):
        if self.handle == 0:
            raise Exception('DBackend: Invalid handle')

        if arg1 is None and arg2 is None:
            # Call the empty constructor
            res = self.py_neon.lib.dBackend_new1(self.handle)
        elif isinstance(arg1, int) and isinstance(arg2, int):
            # Call the constructor with nGpus and runtime
            res = self.py_neon.lib.dBackend_new2(self.handle, arg1, arg2)
        elif isinstance(arg1, list) and isinstance(arg2, int):
            # Call the constructor with devIds and runtime
            from ctypes import c_int, POINTER, byref

            # Convert the list to a ctypes array
            dev_ids = (c_int * len(arg1))(*arg1)
            res = self.py_neon.lib.dBackend_new3(self.handle, dev_ids, arg2)
        else:
            raise Exception('DBackend: Invalid arguments provided')

        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')


    def help_backend_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dBackend_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete backend')

    def __str__(self):
        return ctypes.cast(self.py_neon.lib.get_string(self.handle), ctypes.c_char_p).value.decode('utf-8')
    