"""
Timer Module for Neon Computing Framework

This module provides the Timer class for measuring execution time of
Neon operations with high precision.
"""

import ctypes
from enum import Enum
from typing import List
import warp as wp

import numpy as np

import neon


class Timer(object):
    """
    High-precision timer for measuring Neon operation execution times.
    
    The Timer class provides start/stop functionality for measuring elapsed
    time, useful for benchmarking and performance analysis of Neon computations.
    
    Attributes:
        handle (ctypes.c_void_p): Handle to the C++ timer object.
        nunit (Unit): Time unit for measurements (currently milliseconds).
        neon_gate (neon.Gate): Interface to the C++ Neon library.
    
    Example:
        >>> import neon
        >>> import time
        >>> 
        >>> timer = neon.Timer()
        >>> timer.start()
        >>> 
        >>> # Perform some computation
        >>> time.sleep(0.5)
        >>> 
        >>> elapsed = timer.stop()
        >>> print(f"Elapsed: {elapsed} ms")
        Elapsed: 500.123 ms
        >>> 
        >>> # Can also get time without stopping
        >>> print(timer)  # Uses __str__ method
    """
    
    class Unit(Enum):
        """
        Time unit enumeration.
        
        Attributes:
            sec: Seconds.
            ms: Milliseconds (default).
            us: Microseconds.
        """
        sec = 0
        ms = 0
        us = 1

    def __init__(self, nunit: Unit = Unit.ms):
        """
        Initialize a Timer.
        
        Args:
            nunit (Unit, optional): Time unit for measurements.
                Defaults to Unit.ms (milliseconds).
        
        Raises:
            Exception: If timer initialization fails.
        
        Example:
            >>> timer = neon.Timer()
        """

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

    def start(self) -> None:
        """
        Start the timer.
        
        Begins measuring elapsed time. Call ``stop()`` to get the elapsed duration.
        
        Raises:
            Exception: If the timer fails to start.
        
        Example:
            >>> timer.start()
        """
        res = self.api_start(self.handle)
        if res != 0:
            raise Exception('Failed to start timer')

    def stop(self) -> float:
        """
        Stop the timer and return the elapsed time.
        
        Returns:
            float: Elapsed time in milliseconds since ``start()`` was called.
        
        Example:
            >>> timer.start()
            >>> # ... do work ...
            >>> elapsed_ms = timer.stop()
        """
        res = self.api_stop(self.handle)
        return res

    def time(self) -> float:
        """
        Get the current elapsed time without stopping the timer.
        
        Returns:
            float: Current elapsed time in milliseconds.
        
        Example:
            >>> timer.start()
            >>> # ... do some work ...
            >>> current_ms = timer.time()  # Timer keeps running
            >>> # ... do more work ...
            >>> total_ms = timer.stop()
        """
        res = self.api_time(self.handle)
        return res

    def __str__(self) -> str:
        """Return string representation of elapsed time."""
        return f"{self.time()} ms"


if __name__ == '__main__':
    timer = Timer()
    timer.start()
    import time
    time.sleep(2)
    timer.stop()
    from .logging import logger
    logger.info(str(timer))