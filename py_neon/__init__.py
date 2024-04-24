import copy
import ctypes
from enum import Enum


class PyNeon(object):
    def __init__(self):
        self.handle_type = ctypes.POINTER(ctypes.c_uint64)
        self.lib = ctypes.CDLL('/home/max/repos/neon/warp/neon_warp_testing/neon_py_bindings/cmake-build-debug/libNeonPy/liblibNeonPy.so')
        # # grid_new
        # self.lib.grid_new.argtypes = [self.handle_type]
        # # self.lib.grid_new.re = [ctypes.c_int]
        # # grid_delete
        # self.lib.grid_delete.argtypes = [self.handle_type]
        # # self.lib.grid_delete.restype = [ctypes.c_int]
        # # new_field
        # self.lib.field_new.argtypes = [self.handle_type, self.handle_type]
        # # delete_field
        # self.lib.field_delete.argtypes = [self.handle_type]

    def field_new(self, handle_field: ctypes.c_uint64, handle_grid: ctypes.c_uint64):
        res = self.lib.field_new(handle_field, handle_grid)
        if res != 0:
            raise Exception('Failed to initialize field')

    def field_delete(self, handle_field: ctypes.c_uint64):
        res = self.lib.grid_delete(handle_field)
        if res != 0:
            raise Exception('Failed to initialize grid')


class Execution(Enum):
    device = 1
    host = 2


class Data_view(Enum):
    standard = 0
    internal = 1
    boundary = 2


class Index_3d(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("z", ctypes.c_int)]

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


class DSpan(ctypes.Structure):
    _fields_ = [
        ("dataView", ctypes.c_int),
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

    def get_data_view(self) -> Data_view:
        pass

    def get_data_view(self) -> Data_view:
        pass

    @staticmethod
    def fields_():
        return DSpan._fields_



class DField(object):
    def __init__(self,
                 py_neon: PyNeon,
                 dgrid_handle: ctypes.c_uint64
                 ):
        self.py_neon = py_neon
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.help_load_api()
        self.help_new()

    def __del__(self):
        self.help_delete()

    def help_load_api(self):
        # Importing new functions
        ## new_field
        self.py_neon.lib.dGrid_dField_new.argtypes = [self.handle_type,
                                                      self.handle_type]
        ## delete_field
        self.py_neon.lib.dGrid_dField_delete.argtypes = [self.handle_type]

    def help_new(self):
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        res = self.py_neon.lib.dGrid_dField_new(self.handle)
        if res != 0:
            raise Exception('DGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dGrid_dField_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete field')


class DGrid(object):
    def __init__(self):
        self.py_neon: PyNeon = PyNeon()
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.help_load_api()
        self.help_grid_new()

    def __del__(self):
        if self.handle == 0:
            return
        self.help_grid_delete()

    def help_load_api(self):

        # grid_new
        self.py_neon.lib.dGrid_new.argtypes = [self.py_neon.handle_type]
        # self.lib.grid_new.re = [ctypes.c_int]

        # grid_delete
        self.py_neon.lib.dGrid_delete.argtypes = [self.py_neon.handle_type]
        # self.lib.grid_delete.restype = [ctypes.c_int]

        self.py_neon.lib.dGrid_get_span.argtypes = [self.py_neon.handle_type,
                                                    ctypes.POINTER(DSpan),  # the span object
                                                    ctypes.c_int,  # the execution type
                                                    ctypes.c_int,  # the device id
                                                    ctypes.c_int,  # the data view
                                                    ]

    def help_grid_new(self):
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        res = self.py_neon.lib.dGrid_new(self.handle)
        if res != 0:
            raise Exception('DGrid: Failed to initialize grid')

    def help_grid_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dGrid_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete grid')

    def new_field(self) -> DField:
        field = DField(self.py_neon, self.handle)
        return field

    def get_span(self,
                 execution: Execution,
                 set_idx: int,
                 data_view: Data_view) -> DSpan:
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        span = DSpan()
        ex:int = execution

        res = self.py_neon.lib.dGrid_get_span(self.handle, span, 0, 0, 0)
        if res != 0:
            raise Exception('Failed to get span')
        return span
