import copy
import ctypes
from enum import Enum

from py_neon import Data_view
from py_neon import Index_3d

class DW(ctypes.Structure):
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



class Span(ctypes.Structure):
    _fields_ = [
        ("dataView", DW),
        ("z_ghost_radius", ctypes.c_int),
        ("z_boundary_radius", ctypes.c_int),
        ("max_z_in_domain", ctypes.c_int),
        ("span_dim", Index_3d)
    ]

    def __str__(self):
        str = "<DSpan: addr=%ld>" % (ctypes.addressof(self))
        str += f"\n\tdataView: {self.dataView}"
        str += f"\n\tz_ghost_radius: {self.z_ghost_radius}"
        str += f"\n\tz_boundary_radius: {self.z_boundary_radius}"
        str += f"\n\tmax_z_in_domain: {self.max_z_in_domain}"
        str += f"\n\tspan_dim: {self.span_dim}"
        return str

    def get_span_dim(self):
        return copy.deepcopy(self.span_dim)

    @staticmethod
    def fields_():
        return Span._fields_

