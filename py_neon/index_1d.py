import copy
import ctypes
from enum import Enum

import py_ne

class Index_1d(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32)]

    def __init__(self, x: int):
        self.x = x

    def __str__(self):
        str = "<Index_3d: addr=%ld>" % (ctypes.addressof(self))
        str += f"\n\tx: {self.x}"
        return str

