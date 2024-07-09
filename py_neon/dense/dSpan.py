import copy
import ctypes
from enum import Enum

from py_neon import DataView
from py_neon import Index_3d
from py_neon import Py_neon


class dSpan(ctypes.Structure):
    _fields_ = [
        ("dataView", DataView),
        ("z_ghost_radius", ctypes.c_int),
        ("z_boundary_radius", ctypes.c_int),
        ("max_z_in_domain", ctypes.c_int),
        ("span_dim", Index_3d)
    ]

    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        self.py_neon.lib.dGrid_dSpan_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.py_neon.lib.dGrid_dSpan_get_member_field_offsets.restype = None

    def get_cpp_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = (ctypes.c_size_t * 5)()  # Assuming there are 5 offsets
        self.py_neon.lib.dGrid_dSpan_get_member_field_offsets(offsets, ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]
    


    def __str__(self): 
        str = f"<dSpan: addr={ctypes.addressof(self):#x}>"
        str += f"\n\tdataView: {self.dataView} (offset: {dSpan.dataView.offset})"
        str += f"\n\tz_ghost_radius: {self.z_ghost_radius} (offset: {dSpan.z_ghost_radius.offset})"
        str += f"\n\tz_boundary_radius: {self.z_boundary_radius} (offset: {dSpan.z_boundary_radius.offset})"
        str += f"\n\tmax_z_in_domain: {self.max_z_in_domain} (offset: {dSpan.max_z_in_domain.offset})"
        str += f"\n\tspan_dim: {self.span_dim} (offset: {dSpan.span_dim.offset})"
        return str

    def get_span_dim(self):
        return copy.deepcopy(self.span_dim)

    def get_offsets(self):
        return [dSpan.dataView.offset, dSpan.z_ghost_radius.offset, dSpan.z_boundary_radius.offset, dSpan.max_z_in_domain.offset, dSpan.span_dim.offset]
    
    @staticmethod
    def fields_():
        return dSpan._fields_

