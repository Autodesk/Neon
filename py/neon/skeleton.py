import ctypes
from typing import List
from neon import SkeletonConfig
# from neon import Backend
# #from neon import neon
# from neon import Container
import neon

class Skeleton(object):
    def __init__(self,
                 backend: neon.Backend):

        self.skeleton_handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.backend = backend

        try:
            self.neon_gate: neon.Gate = neon.Gate()
        except Exception as e:
            self.skeleton_handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))

        self.help_load_api()
        self.help_skeleton_new()

    def __del__(self):
        if self.skeleton_handle == 0:
            return
        self.help_skeleton_delete()
        pass

    def help_load_api(self):
        lib_obj = self.neon_gate.lib
        # ------------------------------------------------------------------
        # neon_skeleton_new
        self.api_new = lib_obj.neon_skeleton_new
        self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                 self.neon_gate.handle_type]
        self.api_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_delete
        self.api_delete = lib_obj.neon_skeleton_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_sequence
        self.api_sequence = lib_obj.neon_skeleton_sequence
        self.api_sequence.argtypes = [self.neon_gate.handle_type,
                                      ctypes.c_char_p,
                                      ctypes.c_int,
                                      ctypes.POINTER(self.neon_gate.handle_type),
                                      ctypes.c_int]
        self.api_sequence.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_run
        self.api_run = lib_obj.neon_skeleton_run
        self.api_run.argtypes = [self.neon_gate.handle_type]
        self.api_run.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_ioToDot
        self.api_ioToDot = lib_obj.neon_skeleton_ioToDot
        self.api_ioToDot.argtypes = [self.neon_gate.handle_type,
                                 ctypes.c_char_p,
                                 ctypes.c_char_p,
                                 ctypes.c_int]
        self.api_ioToDot.restype = ctypes.c_int

    def help_skeleton_new(self):
        if self.skeleton_handle.value != ctypes.c_void_p(0).value:
            raise Exception(f'Skeleton: Invalid handle {self.skeleton_handle}')

        res = self.api_new(ctypes.pointer(self.skeleton_handle),
                                                 self.backend.backend_handle)

        if res != 0:
            raise Exception('Backend: Failed to initialize backend')

    def help_skeleton_delete(self):
        if self.skeleton_handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.skeleton_handle))
        if res != 0:
            raise Exception('Failed to delete backend')

    def sequence(self, name: str, containers: List[neon.Container], occ: neon.SkeletonConfig.OCC = neon.SkeletonConfig.OCC.none()
    ):
        self.containers = containers
        self.handle_list = (ctypes.c_void_p * len(containers))()
        for i in range(len(self.handle_list)):
            self.handle_list[i] = containers[i].container_handle
        from .logging import logger
        logger.debug(f"handle_list {self.handle_list}")
        logger.debug(f"handle_list[0] {hex(self.handle_list[0])}")
        self.api_sequence(self.skeleton_handle,
                          name.encode('utf-8'),
                          len(self.handle_list),
                          self.handle_list,
                          occ.value)

    def run(self):
        self.api_run(self.skeleton_handle)

    def ioToDot(
            self,
            filename: str,
            graph_name: str
    ):
        self.api_ioToDot(self.skeleton_handle,
                         filename.encode('utf-8'),
                         graph_name.encode('utf-8'),
                         0
                         )
