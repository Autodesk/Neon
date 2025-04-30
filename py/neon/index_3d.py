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


    @staticmethod
    def warp_register_builtins():
        import warp as wp

        # register type
        wp.types.add_type(Index_3d, native_name="NeonIndex3d")

        # create dense index
        wp.context.add_builtin(
            "neon_idx_3d",
            input_types={"x": int, "y": int, "z": int},
            value_type=Index_3d,
            missing_grad=True,
        )

        # create dense index
        wp.context.add_builtin(
            "neon_init",
            input_types={"idx":Index_3d, "x": int, "y": int, "z": int},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_x",
            input_types={"idx":Index_3d},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_y",
            input_types={"idx":Index_3d},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_z",
            input_types={"idx":Index_3d},
            value_type=int,
            missing_grad=True,
        )

        # print dense index
        wp.context.add_builtin(
            "neon_print",
            input_types={"a": Index_3d},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_is_equal",
            input_types={"idx": Index_3d, "x": int, "y": int, "z" : int},
            value_type=bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_cuda_info",
            input_types={},
            value_type=None,
            missing_grad=True,
        )