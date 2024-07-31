import ctypes
from enum import Enum


class DataView(ctypes.Structure):
    _fields_ = [("data_view", ctypes.c_char)]

    class Values(Enum):
        standard = 0
        internal = 1
        boundary = 2


    def __init__(self, data_view: 'DataView.Values'):
        if data_view == DataView.Values.standard:
            self.data_view = ctypes.c_char(b'\x00')
        elif data_view == DataView.Values.internal:
            self.data_view = ctypes.c_char(b'\x01')
        elif data_view == DataView.Values.boundary:
            self.data_view = ctypes.c_char(b'\x02')

    def __str__(self):
        str_repr = "<DDDData_view: addr=%ld, sizeof %ld>" % (ctypes.addressof(self), ctypes.sizeof(self))
        if self.data_view == ctypes.c_char(b'\x00'):
            str_repr += f"\n\tdataView: {'standard'}"
        elif self.data_view == ctypes.c_char(b'\x01'):
            str_repr += f"\n\tdataView: {'internal'}"
        elif self.data_view == ctypes.c_char(b'\x02'):
            str_repr += f"\n\tdataView: {'boundary'}"
        return str_repr

    @property
    def value(self):
        return self.data_view

    @value.setter
    def value(self, data_view: Values):
        self.data_view = data_view

    @staticmethod
    def standard():
        return DataView(DataView.Values.standard)

    @staticmethod
    def internal():
        return DataView(DataView.Values.internal)

    @staticmethod
    def boundary():
        return DataView(DataView.Values.boundary)

    @staticmethod
    def from_int(v: int):
        if v == 0:
            return DataView(DataView.Values.standard)
        if v == 1:
            return DataView(DataView.Values.internal)
        if v == 2:
            return DataView(DataView.Values.boundary)
        # rise exeption
        raise Exception('Invalid DataView value')


    def __eq__(self, other):
        if not isinstance(other, DataView):
            return NotImplemented
        return self.data_view == other.data_view
