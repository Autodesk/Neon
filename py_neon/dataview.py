import copy
import ctypes
from enum import Enum

class DataView(ctypes.Structure):
    _fields_ = [("data_view", ctypes.c_uint8)]

    class Values(Enum):
        standard = 0
        internal = 1
        boundary = 2

    def __init__(self, data_view: Values):
        if data_view == DataView.Values.standard:
            self.data_view = 0
        if data_view == DataView.Values.internal:
            self.data_view = 1
        if data_view == DataView.Values.boundary:
            self.data_view = 2

    @property
    def value(self):
        return self.data_view

    @value.setter
    def value(self, data_view: Values):
        self.data_view = data_view

    @staticmethod
    def standard() :
        return DataView(DataView.Values.standard)

    @staticmethod
    def internal():
        return DataView(DataView.Values.internal)

    @staticmethod
    def boundary():
        return DataView(DataView.Values.boundary)

    def __str__(self):
        str = "<Data_view: addr=%ld, sizeof %ld>" % (ctypes.addressof(self), ctypes.sizeof(self))
        if self.value == DataView.Values.standard:
            str += f"\n\tdataView: {'standard'}"
        if self.value == DataView.Values.internal:
            str += f"\n\tdataView: {'internal'}"
        if self.value == DataView.Values.boundary:
            str += f"\n\tdataView: {'boundary'}"
        return str
    
    def __eq__(self, other):
        if not isinstance(other, DataView):
            return NotImplemented
        return self.data_view == other.data_view

