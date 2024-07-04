import ctypes
from enum import Enum
from typing import List

import numpy as np

from py_neon import Py_neon


class Backend(object):
    class Runtime(Enum):
        none = 0
        system = 0
        stream = 1
        openmp = 2

    def __init__(self,
                 runtime: Runtime = Runtime.openmp,
                 n_dev: int = 1,
                 dev_idx_list: List[int] = [0]):

        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_backend_new(runtime, n_dev, dev_idx_list)

    def __del__(self):
        if self.handle == 0:
            return
        self.help_backend_delete()

    def help_load_api(self):

        # # backend_new
        # self.py_neon.lib.dBackend_new1.argtypes = [self.py_neon.handle_type]
        # self.py_neon.lib.dBackend_new1.restype = ctypes.c_int
        #
        # # backend_new
        # self.py_neon.lib.dBackend_new2.argtypes = [self.py_neon.handle_type,
        #                                            ctypes.c_int,
        #                                            ctypes.c_int]
        # self.py_neon.lib.dBackend_new2.restype = ctypes.c_int

        # backend_new
        self.py_neon.lib.dBackend_new.argtypes = [self.py_neon.handle_type,
                                                  ctypes.c_int,
                                                  ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_int)]

        self.py_neon.lib.dBackend_new.restype = ctypes.c_int

        # backend_delete
        self.py_neon.lib.dBackend_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_delete.restype = ctypes.c_int

        # backend_get_string
        self.py_neon.lib.dBackend_get_string.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_get_string.restype = ctypes.c_char_p

        # TODOMATT get num devices
        # TODOMATT get device type

    def help_backend_new(self,
                         runtime: Runtime,
                         n_dev: int,
                         dev_idx_list: List[int]):
        if self.handle.value != ctypes.c_uint64(0).value:
            raise Exception(f'DBackend: Invalid handle {self.handle}')

        if n_dev > len(dev_idx_list):
            dev_idx_list = list(range(n_dev))
        else:
            n_dev = len(dev_idx_list)

        dev_idx_np = np.array(dev_idx_list, dtype=int)
        dev_idx_ptr = dev_idx_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        res = self.py_neon.lib.dBackend_new(ctypes.byref(self.handle),
                                            runtime.value,
                                            n_dev,
                                            dev_idx_ptr)
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
