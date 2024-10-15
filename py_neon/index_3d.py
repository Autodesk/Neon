import ctypes
import typing


class Index_3d(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32),
                ("y", ctypes.c_int32),
                ("z", ctypes.c_int32)]

    def __init__(self,
                 x: int,
                 y: int,
                 z: int):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z
        raise IndexError("Index out of range")

    def to_wp_kernel_dim(self) -> typing.Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def __str__(self):
        str = f"({self.x}, "
        str += f"{self.y}, "
        str += f"{self.z})"
        str += "<Index_3d: addr=%ld>" % (ctypes.addressof(self))

        return str

    def __eq__(self, other):
        if not isinstance(other, Index_3d):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)
