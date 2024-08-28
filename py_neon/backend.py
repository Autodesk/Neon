import ctypes
from enum import Enum
from typing import List
import warp as wp

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

        self.backend_handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.cuda_driver_handle: ctypes.c_void_p = ctypes.c_void_p(0)

        self.n_dev = n_dev
        self.dev_idx_list = dev_idx_list
        self.runtime = runtime

        devices = {}
        devices['cuda:0'] = wp.get_device("cuda:0")
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.backend_handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_backend_new()

    def __del__(self):
        if self.backend_handle == 0:
            return
        self.help_backend_delete()
        pass

    def help_load_api(self):
        # ------------------------------------------------------------------
        # backend_new
        lib_obj = self.py_neon.lib
        self.api_new = lib_obj.dBackend_new
        self.api_new.argtypes = [ctypes.POINTER(self.py_neon.handle_type),
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int)]
        self.api_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # backend_delete
        self.py_neon.lib.dBackend_delete.argtypes = [ctypes.POINTER(self.py_neon.handle_type)]
        self.py_neon.lib.dBackend_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # backend_get_string
        self.py_neon.lib.dBackend_get_string.argtypes = [self.py_neon.handle_type]
        self.py_neon.lib.dBackend_get_string.restype = ctypes.c_char_p
        # ------------------------------------------------------------------
        # cuda_driver_new
        self.py_neon.lib.cuda_driver_new.argtypes = [ctypes.POINTER(self.py_neon.handle_type),
                                                     self.py_neon.handle_type]
        self.py_neon.lib.cuda_driver_new.restype = ctypes.c_int

        # self.py_neon.lib.cuda_driver_new.argtypes = [self.py_neon.handle_type,
        #                                              self.py_neon.handle_type]
        # self.py_neon.lib.cuda_driver_new.restype = None
        # ------------------------------------------------------------------
        # cuda_driver_delete
        self.py_neon.lib.cuda_driver_delete.argtypes = [ctypes.POINTER(self.py_neon.handle_type)]
        self.py_neon.lib.cuda_driver_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------

        # TODOMATT get num devices
        # TODOMATT get device type

    def help_backend_new(self):
        if self.backend_handle.value != ctypes.c_void_p(0).value:
            raise Exception(f'DBackend: Invalid handle {self.backend_handle}')

        if self.n_dev > len(self.dev_idx_list):
            self.dev_idx_list = list(range(self.n_dev))
        else:
            self.n_dev = len(self.dev_idx_list)

        # Loading the device list into a contiguous array
        dev_array = (ctypes.c_int * self.n_dev)(*self.dev_idx_list)

        res = self.py_neon.lib.dBackend_new(ctypes.pointer(self.backend_handle),
                                            self.runtime.value,
                                            self.n_dev,
                                            dev_array)

        print(f"NEON PYTHON self.backend_handle: {hex(self.backend_handle.value)}")
        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')

        self.py_neon.lib.cuda_driver_new(ctypes.pointer(self.cuda_driver_handle),
                                         self.backend_handle)
        pass


    def help_backend_delete(self):
        if self.backend_handle == 0:
            return
        # print(f'PYTHON cuda_driver_handle {hex(self.cuda_driver_handle.value)}')
        self.py_neon.lib.cuda_driver_delete(ctypes.pointer(self.cuda_driver_handle))
        # print(f'PYTHON backend_handle {hex(self.backend_handle.value)}')
        res = self.py_neon.lib.dBackend_delete(ctypes.pointer(self.backend_handle))
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
        return self.py_neon.lib.dBackend_sync(self.backend_handle)


    def get_device_name(self, dev_idx: int):
        if self.runtime == Backend.Runtime.stream:
            dev_id = self.dev_idx_list[dev_idx]
            return f"cuda:{dev_id}"
        else:
            dev_id = self.dev_idx_list[dev_idx]
            return f"cpu:{dev_id}"
