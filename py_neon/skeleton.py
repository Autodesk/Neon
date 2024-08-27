import ctypes
from typing import List

from py_neon import Backend
from py_neon import Py_neon
from wpne import Container


class Skeleton(object):
    def __init__(self,
                 backend: Backend):

        self.skeleton_handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.backend = backend

        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.skeleton_handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))

        self.help_load_api()
        self.help_skeleton_new()

    def __del__(self):
        if self.skeleton_handle == 0:
            return
        self.help_skeleton_new()
        pass

    def help_load_api(self):
        lib_obj = self.py_neon.lib
        # ------------------------------------------------------------------
        # warp_skeleton_new
        self.api_new = lib_obj.warp_skeleton_new
        self.api_new.argtypes = [ctypes.POINTER(self.py_neon.handle_type),
                                 self.py_neon.handle_type]
        self.api_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # warp_skeleton_delete
        self.api_delete = lib_obj.warp_skeleton_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.py_neon.handle_type)]
        self.api_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # warp_skeleton_sequence
        self.api_sequence = lib_obj.warp_skeleton_sequence
        self.api_sequence.argtypes = [self.py_neon.handle_type,
                                      ctypes.c_char_p,
                                      ctypes.c_int,
                                      ctypes.POINTER(self.py_neon.handle_type)]
        self.api_sequence.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # warp_skeleton_run
        self.api_run = lib_obj.warp_skeleton_run
        self.api_run.argtypes = [self.py_neon.handle_type]
        self.api_run.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # warp_skeleton_ioToDot
        self.api_run = lib_obj.warp_skeleton_ioToDot
        self.api_run.argtypes = [self.py_neon.handle_type,
                                 ctypes.c_char_p,
                                 ctypes.c_char_p,
                                 ctypes.c_int]
        self.api_run.restype = ctypes.c_int

    def help_skeleton_new(self):
        if self.skeleton_handle.value != ctypes.c_void_p(0).value:
            raise Exception(f'Skeleton: Invalid handle {self.skeleton_handle}')

        res = self.py_neon.lib.warp_skeleton_new(ctypes.pointer(self.skeleton_handle),
                                                 self.backend.backend_handle)

        if res != 0:
            raise Exception('Backend: Failed to initialize backend')

    def help_skeleton_delete(self):
        if self.skeleton_handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.skeleton_handle))
        if res != 0:
            raise Exception('Failed to delete backend')

    def sequence(self, name: str, containers: List[Container]):
        self.containers = containers
        self.handle_list = (ctypes.c_void_p * len(containers))()
        for i in range(len(self.handle_list)):
            self.handle_list[i] = containers[i].container_handle
        print(f"PYTHON handle_list {self.handle_list}")
        print(f"PYTHON handle_list[0] {hex(self.handle_list[0])}")
        self.api_sequence(self.skeleton_handle,
                          name.encode('utf-8'),
                          len(self.handle_list),
                          self.handle_list)

    def run(self):
        self.api_run(self.skeleton_handle)