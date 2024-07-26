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

        self.backend_handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.cuda_driver_handle: ctypes.c_uint64 = ctypes.c_uint64(0)

        self.n_dev = n_dev
        self.dev_idx_list = dev_idx_list
        self.runtime = runtime

        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.backend_handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_backend_new()

    def __del__(self):
        if self.backend_handle == 0:
            return
        self.help_backend_delete()

    def help_load_api(self):
        # ------------------------------------------------------------------
        # backend_new
        self.py_neon.lib.dBackend_new.argtypes = [self.py_neon.handle_type,
                                                  ctypes.c_int,
                                                  ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_int)]
        self.py_neon.lib.dBackend_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # backend_delete
        self.py_neon.lib.dBackend_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # backend_get_string
        self.py_neon.lib.dBackend_get_string.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_get_string.restype = ctypes.c_char_p
        # ------------------------------------------------------------------
        # cuda_driver_new
        self.py_neon.lib.cuda_driver_new.argtypes = [self.py_neon.handle_type,
                                                     self.py_neon.handle_type, ]
        self.py_neon.lib.cuda_driver_new.restype = None
        # ------------------------------------------------------------------
        # cuda_driver_delete
        self.py_neon.lib.cuda_driver_delete.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.cuda_driver_delete.restype = None
        # ------------------------------------------------------------------

        # TODOMATT get num devices
        # TODOMATT get device type

    def help_backend_new(self):
        if self.backend_handle.value != ctypes.c_uint64(0).value:
            raise Exception(f'DBackend: Invalid handle {self.backend_handle}')

        if self.n_dev > len(self.dev_idx_list):
            dev_idx_list = list(range(self.n_dev))
        else:
            self.n_dev = len(self.dev_idx_list)

        dev_idx_np = np.array(self.dev_idx_list, dtype=int)
        dev_idx_ptr = dev_idx_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        res = self.py_neon.lib.dBackend_new(ctypes.byref(self.backend_handle),
                                            self.runtime.value,
                                            self.n_dev,
                                            dev_idx_ptr)
        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')

        self.py_neon.lib.cuda_driver_new(self.cuda_driver_handle,
                                         self.backend_handle)

    def help_backend_delete(self):
        if self.backend_handle == 0:
            return
        self.py_neon.lib.cuda_driver_delete(self.cuda_driver_handle)
        res = self.py_neon.lib.dBackend_delete(self.backend_handle)
        if res != 0:
            raise Exception('Failed to delete backend')

    def get_num_devices(self):
        return self.n_dev

    def get_warp_device_name(self):
        if self.runtime == Backend.Runtime.stream:
            return 'cuda'
        else:
            return 'cpu'

    def __str__(self):
        return ctypes.cast(self.py_neon.lib.get_string(self.backend_handle), ctypes.c_char_p).value.decode('utf-8')

    def sync(self):
        return self.py_neon.lib.dBackend_sync(ctypes.byref(self.backend_handle))
