import ctypes
from enum import Enum


class DataUse(ctypes.Structure):
    _fields_ = [("data_use", ctypes.c_char)]

    class Values(Enum):
        HOST_DEVICE = 0
        DEVICE = 1
        HOST = 2


    def __init__(self, data_use: 'DataUse.Values'):
        if data_use == DataUse.Values.HOST_DEVICE:
            self.data_use = ctypes.c_char(b'\x00')
        elif data_use == DataUse.Values.DEVICE:
            self.data_use = ctypes.c_char(b'\x01')
        elif data_use == DataUse.Values.HOST:
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
    def host_device():
        return DataUse(DataUse.Values.HOST_DEVICE)

    @staticmethod
    def device():
        return DataUse(DataUse.Values.DEVICE)

    @staticmethod
    def host():
        return DataUse(DataUse.Values.HOST_DEVICE)

    @staticmethod
    def from_int(v: int):
        if v == 0:
            return DataUse(DataUse.Values.HOST_DEVICE)
        if v == 1:
            return DataUse(DataUse.Values.DEVICE)
        if v == 2:
            return DataUse(DataUse.Values.HOST)
        # rise exeption
        raise Exception('Invalid DataUse value')


    def __eq__(self, other):
        if not isinstance(other, DataUse):
            return NotImplemented
        return self.data_use == other.data_use

