import copy
import ctypes
from enum import Enum

#import py_ne


class Execution(ctypes.Structure):
    _fields_ = [("execution", ctypes.c_uint8)]

    class Values(Enum):
        device = 0
        host = 1

    def __init__(self, execution: Values):
        if execution == Execution.Values.device:
            self.execution = 0
        if execution == Execution.Values.host:
            self.execution = 1

    def __int__(self):
        return self.execution

    @property
    def value(self):
        return self.execution

    @value.setter
    def value(self, execution: Values):
        self.execution = execution

    @staticmethod
    def device():
        return Execution(Execution.Values.device)

    @staticmethod
    def host():
        return Execution(Execution.Values.host)
