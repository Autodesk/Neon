import ctypes


class PyNeon(object):
    def __init__(self):
        self.handle_type = ctypes.POINTER(ctypes.c_uint64)
        self.lib = ctypes.CDLL('/home/max/repos/neon/warp/py_bindings/cmake-build-debug/libNeonPy/liblibNeonPy.so')
        # grid_new
        self.lib.grid_new.argtypes = [self.handle_type]
        # self.lib.grid_new.re = [ctypes.c_int]
        # grid_delete
        self.lib.grid_delete.argtypes = [self.handle_type]
        # self.lib.grid_delete.restype = [ctypes.c_int]
        # new_field
        self.lib.field_new.argtypes = [self.handle_type, self.handle_type]
        # delete_field
        self.lib.field_delete.argtypes = [self.handle_type]

    def grid_new(self, grid_handle: ctypes.c_uint64):
        res = self.lib.grid_new(grid_handle)
        if res != 0:
            raise Exception('Failed to initialize grid')

    def grid_delete(self, handle: ctypes.c_uint64):
        res = self.lib.grid_delete(handle)
        if res != 0:
            raise Exception('Failed to initialize grid')

    def field_new(self, handle_field: ctypes.c_uint64, handle_grid: ctypes.c_uint64):
        res = self.lib.field_new(handle_field, handle_grid)
        if res != 0:
            raise Exception('Failed to initialize field')

    def field_delete(self, handle_field: ctypes.c_uint64):
        res = self.lib.grid_delete(handle_field)
        if res != 0:
            raise Exception('Failed to initialize grid')


def get_lib(self):
    return self.lib


class Field(object):
    def __init__(self,
                 handle: ctypes.c_uint64,
                 py_neon: PyNeon):
        self.handle = handle
        self.py_neon = py_neon

    def get_lib(self):
        return self.lib

    def io_to_vtk(self):
        pass


class Grid(object):
    def __init__(self):
        self.py_neon: PyNeon = PyNeon()
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.py_neon.grid_new(grid_handle = self.handle)

    def new_field(self) -> Field:
        handle_field: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.py_neon.field_new(handle_field, self.handle)
