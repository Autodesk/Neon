import ctypes
from enum import Enum
from typing import List
import warp as wp

import numpy as np

import neon


class Timer(object):
    class Unit(Enum):
        sec = 0
        ms = 0
        us = 1

    def __init__(self,
                 nunit: Unit = Unit.ms):

        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.nunit = nunit

        try:
            self.neon_gate: neon = neon.Gate()
        except Exception as e:
            self.handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_new()

    def __del__(self):
        if self.handle == 0:
            return
        self.help_delete()
        pass

    def help_load_api(self):
        # ------------------------------------------------------------------
        # timer_ms_new
        lib_obj = self.neon_gate.lib
        self.api_new = lib_obj.timer_ms_new
        self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # timer_ms_delete
        self.api_delete = lib_obj.timer_ms_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # timer_ms_start
        self.api_start = lib_obj.timer_ms_start
        self.api_start.argtypes = [self.neon_gate.handle_type]
        self.api_start.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # timer_ms_stop
        self.api_stop = lib_obj.timer_ms_stop
        self.api_stop.argtypes = [self.neon_gate.handle_type]
        self.api_stop.restype = ctypes.c_double
        # ------------------------------------------------------------------
        # timer_ms_time
        self.api_time = lib_obj.timer_ms_time
        self.api_time.argtypes = [self.neon_gate.handle_type]
        self.api_time.restype = ctypes.c_double
        # ------------------------------------------------------------------

    def help_new(self):
        
        res = self.api_new(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete backend')

    def start(self):
        res = self.api_start(self.handle)
        if res != 0:
            raise Exception('Failed to start timer')

    def stop(self):
        res = self.api_stop(self.handle)
        return res

    def time(self):
        res = self.api_time(self.handle)
        return res

    def __str__(self):
        return f"{self.time()} ms"


if __name__ == '__main__':
    # Create a timer
    timer = Timer()
    timer.start()
    # Do something (i.e. sleep for 2 seconds)
    import time
    time.sleep(2)
    timer.stop()
    from .logging import logger
    logger.info(str(timer))