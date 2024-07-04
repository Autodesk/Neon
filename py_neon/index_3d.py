import copy
import ctypes
from enum import Enum

import py_ne

class Index_3d(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32), ("z", ctypes.c_int32)]

    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        str = "<Index_3d: addr=%ld>" % (ctypes.addressof(self))
        str += f"\n\tx: {self.x}"
        str += f"\n\ty: {self.y}"
        str += f"\n\tz: {self.z}"
        return str
    
    def __eq__(self, other):
        if not isinstance(other, Index_3d):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)

