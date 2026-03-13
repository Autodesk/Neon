"""
Execution Module for Neon Computing Framework

This module provides the Execution class, which specifies where computational
kernels should be executed (device/GPU or host/CPU).
"""

import copy
import ctypes
from enum import Enum


class Execution(ctypes.Structure):
    """
    Specifies the execution target for Neon operations.
    
    The Execution class determines whether kernels run on the GPU (device)
    or CPU (host). This is a ctypes Structure that can be passed directly
    to C++ functions.
    
    Attributes:
        execution (ctypes.c_uint8): Internal value (0=device, 1=host).
    
    Example:
        >>> import neon
        >>> 
        >>> # Create device (GPU) execution context
        >>> gpu_exec = neon.Execution.device()
        >>> 
        >>> # Create host (CPU) execution context
        >>> cpu_exec = neon.Execution.host()
        >>> 
        >>> # Use with container
        >>> container = Container(name="my_kernel", 
        ...                       loading_lambda=loader,
        ...                       execution=neon.Execution.device())
    
    Note:
        Use the static factory methods ``device()`` and ``host()`` to create
        instances rather than the constructor directly.
    """
    
    _fields_ = [("execution", ctypes.c_uint8)]

    class Values(Enum):
        """
        Enumeration of execution targets.
        
        Attributes:
            device: Execute on GPU.
            host: Execute on CPU.
        """
        device = 0
        host = 1

    def __init__(self, execution: Values):
        """
        Initialize an Execution context.
        
        Args:
            execution (Values): The execution target (device or host).
        
        Note:
            Prefer using ``Execution.device()`` or ``Execution.host()`` factory methods.
        """
        if execution == Execution.Values.device:
            self.execution = 0
        if execution == Execution.Values.host:
            self.execution = 1

    def __int__(self) -> int:
        """Return the integer value of this execution context."""
        return self.execution

    @property
    def value(self) -> int:
        """Get the execution target value."""
        return self.execution

    @value.setter
    def value(self, execution: Values) -> None:
        """Set the execution target value."""
        self.execution = execution

    @staticmethod
    def device() -> 'Execution':
        """
        Create a device (GPU) execution context.
        
        Returns:
            Execution: An Execution instance configured for GPU execution.
        
        Example:
            >>> exec_ctx = neon.Execution.device()
        """
        return Execution(Execution.Values.device)

    @staticmethod
    def host() -> 'Execution':
        """
        Create a host (CPU) execution context.
        
        Returns:
            Execution: An Execution instance configured for CPU execution.
        
        Example:
            >>> exec_ctx = neon.Execution.host()
        """
        return Execution(Execution.Values.host)
