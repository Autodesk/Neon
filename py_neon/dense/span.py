import copy
import ctypes
from enum import Enum

from py_neon import DataView
from py_neon import Index_3d


class Span(ctypes.Structure):
    _fields_ = [
        ("dataView", DataView),
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

