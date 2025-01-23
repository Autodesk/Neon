import ctypes
import typing
import neon.index_3d
import warp as wp

class bIndex(ctypes.Structure):
    _fields_ = [("location", neon.Index_3d)]

    def __init__(self,
                 val: neon.index_3d.Index_3d):
        self.location = val

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 0:
            return self.location.x
        if index == 1:
            return self.location.y
        if index == 2:
            return self.location.z
        raise IndexError("Index out of range")

    def __str__(self):
        str = f"({self.location.x}, "
        str += f"{self.location.y}, "
        str += f"{self.location.z})"
        str += "<Index_3d: addr=%ld>" % (ctypes.addressof(self))

        return str

    def __eq__(self, other):
        if not isinstance(other, bIndex):
            return NotImplemented
        return self.location == other.location

    @staticmethod
    def register_builtins():

        # register type
        wp.types.add_type(bIndex, native_name="NeonBlockIdx")

