import copy
import ctypes
from enum import Enum

class Data_view(ctypes.Structure):
    _fields_ = [("data_view", ctypes.c_uint8)]

    class Values(Enum):
        standard = 0
        internal = 1
        boundary = 2

    def __init__(self, data_view: Values):
        if data_view == Data_view.Values.standard:
            self.data_view = 1
        if data_view == Data_view.Values.internal:
            self.data_view = 2
        if data_view == Data_view.Values.boundary:
            self.data_view = 3

    @property
    def value(self):
        return self.data_view

    @value.setter
    def value(self, data_view: Values):
        self.data_view = data_view

    @staticmethod
    def standard() :
        return Data_view(Data_view.Values.standard)

    @staticmethod
    def internal():
        return Data_view(Data_view.Values.internal)

    @staticmethod
    def boundary():
        return Data_view(Data_view.Values.boundary)

    def __str__(self):
        str = "<Data_view: addr=%ld, sizeof %ld>" % (ctypes.addressof(self), ctypes.sizeof(self))
        if self.value == Data_view.Values.standard:
            str += f"\n\tdataView: {'standard'}"
        if self.value == Data_view.Values.internal:
            str += f"\n\tdataView: {'internal'}"
        if self.value == Data_view.Values.boundary:
            str += f"\n\tdataView: {'boundary'}"
        return str

