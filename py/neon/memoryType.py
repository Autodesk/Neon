"""
MemoryType Module for Neon Computing Framework

This module provides the MemoryType class, which specifies where field
data should be allocated (host memory, device memory, or both).
"""

import ctypes
from enum import Enum


class MemoryType(ctypes.Structure):
    """
    Specifies memory allocation location for Neon fields.
    
    MemoryType determines where field data is allocated. This affects
    data accessibility and transfer requirements between host and device.
    
    Memory types:
        - ``HOST_DEVICE``: Data on both host and device (mirrored)
        - ``DEVICE``: Data only on GPU device memory
        - ``HOST``: Data only on CPU host memory
    
    This is a ctypes Structure that can be passed directly to C++ functions.
    
    Example:
        >>> import neon
        >>> 
        >>> # Allocate field on both host and device
        >>> field = grid.new_field(
        ...     cardinality=1,
        ...     dtype=wp.float32,
        ...     memory_type=neon.MemoryType.host_device()
        ... )
        >>> 
        >>> # Allocate field only on device (GPU)
        >>> field = grid.new_field(
        ...     cardinality=1,
        ...     dtype=wp.float32,
        ...     memory_type=neon.MemoryType.device()
        ... )
    
    Note:
        Use the static factory methods to create instances.
    """
    
    _fields_ = [("data_use", ctypes.c_char)]

    class Values(Enum):
        """
        Enumeration of memory location options.
        
        Attributes:
            HOST_DEVICE: Allocate on both host and device (mirrored).
            DEVICE: Allocate only on GPU device.
            HOST: Allocate only on CPU host.
        """
        HOST_DEVICE = 0
        DEVICE = 1
        HOST = 2

    def __init__(self, data_use: 'MemoryType.Values'):
        """
        Initialize a MemoryType.
        
        Args:
            data_use (Values): The memory location type.
        
        Note:
            Prefer using ``MemoryType.host_device()``, ``MemoryType.device()``,
            or ``MemoryType.host()`` factory methods.
        """
        if data_use == MemoryType.Values.HOST_DEVICE:
            self.data_use = ctypes.c_char(b'\x00')
        elif data_use == MemoryType.Values.DEVICE:
            self.data_use = ctypes.c_char(b'\x01')
        elif data_use == MemoryType.Values.HOST:
            self.data_use = ctypes.c_char(b'\x02')

    def __str__(self):
        str_repr = "<DDDdata_use: addr=%ld, sizeof %ld>" % (ctypes.addressof(self), ctypes.sizeof(self))
        if self.data_use == ctypes.c_char(b'\x00'):
            str_repr += f"\n\tDataUse: {'HOST_DEVICE'}"
        elif self.data_use == ctypes.c_char(b'\x01'):
            str_repr += f"\n\tDataUse: {'DEVICE'}"
        elif self.data_use == ctypes.c_char(b'\x02'):
            str_repr += f"\n\tDataUse: {'HOST'}"
        return str_repr

    @property
    def value(self):
        return self.data_use

    @value.setter
    def value(self, data_use: Values):
        self.data_use = data_use

    @staticmethod
    def host_device() -> 'MemoryType':
        """
        Create a MemoryType for host and device allocation.
        
        Data will be mirrored on both CPU and GPU, allowing access from either.
        
        Returns:
            MemoryType: A MemoryType for host+device allocation.
        
        Example:
            >>> mt = neon.MemoryType.host_device()
        """
        return MemoryType(MemoryType.Values.HOST_DEVICE)

    @staticmethod
    def device() -> 'MemoryType':
        """
        Create a MemoryType for device-only allocation.
        
        Data will only exist on GPU memory.
        
        Returns:
            MemoryType: A MemoryType for device-only allocation.
        
        Example:
            >>> mt = neon.MemoryType.device()
        """
        return MemoryType(MemoryType.Values.DEVICE)

    @staticmethod
    def host() -> 'MemoryType':
        """
        Create a MemoryType for host-only allocation.
        
        Data will only exist on CPU memory.
        
        Returns:
            MemoryType: A MemoryType for host-only allocation.
        
        Example:
            >>> mt = neon.MemoryType.host()
        """
        return MemoryType(MemoryType.Values.HOST_DEVICE)

    @staticmethod
    def from_int(v: int) -> 'MemoryType':
        """
        Create a MemoryType from an integer value.
        
        Args:
            v (int): 0 for HOST_DEVICE, 1 for DEVICE, 2 for HOST.
        
        Returns:
            MemoryType: The corresponding MemoryType.
        
        Raises:
            Exception: If v is not 0, 1, or 2.
        """
        if v == 0:
            return MemoryType(MemoryType.Values.HOST_DEVICE)
        if v == 1:
            return MemoryType(MemoryType.Values.DEVICE)
        if v == 2:
            return MemoryType(MemoryType.Values.HOST)
        raise Exception('Invalid DataUse value')


    def __eq__(self, other):
        if not isinstance(other, MemoryType):
            return NotImplemented
        return self.data_use == other.data_use

