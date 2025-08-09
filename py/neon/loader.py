from enum import Enum
import neon

class Loader:
    class Access(Enum):
        read = 1
        write = 2

    class Operation(Enum):
        map = 0
        stencil = 1
        stencil_up = 2
        stencil_down = 3

    class Discretization(Enum):
        cartesian = 0
        lattice = 1

    class Token:
        def __init__(self, field, access, operation, discretization):
            self.field = field
            self.access = access
            self.operation = operation
            self.discretization = discretization
        def get_field(self):
            return self.field
        def get_access(self):
            return self.access
        def get_operation(self):
            return self.operation
        def get_discretization(self):
            return self.discretization

    def __init__(self,
                 execution: neon.Execution,
                 gpu_id: int,
                 data_view: neon.DataView,
                 parsing: bool = False):

        self.mres_level = None
        self.execution = execution
        self.gpu_id = gpu_id
        self.data_view = data_view
        self.parsing = parsing

        self.tokens = []

        self.kernel = None
        self.neon_grid = None


    def get_read_handle(self, neon_field,
                        operation : Operation = Operation.map,
                        discretization: Discretization = Discretization.cartesian):
        if self.parsing:
            access = Loader.Access.read
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)
        return partition

    def get_write_handle(self,
                         neon_field,
                         operation: Operation = Operation.map,
                         discretization:Discretization = Discretization.cartesian):
        if self.parsing:
            access = Loader.Access.write
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)

        return partition

    def get_mres_write_handle(self,
                         neon_field,
                         operation: Operation = Operation.map,
                         discretization:Discretization = Discretization.cartesian):

        if self.parsing:
            access = Loader.Access.write
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.mres_level,
            self.execution,
            self.gpu_id,
            self.data_view)

        return partition

    def get_mres_read_handle(
            self,
            neon_field,
            operation: Operation = Operation.map,
            discretization: Discretization = Discretization.cartesian):

        if self.parsing:
            access = Loader.Access.read
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.mres_level,
            self.execution,
            self.gpu_id,
            self.data_view
        )

        return partition

    def set_grid(self, grid):
        self.neon_grid = grid

    def set_mres_grid(self, grid, level):
        self.neon_grid = grid
        self.mres_level = level

    def get_mres_level(self):
        return self.mres_level

    def _retrieve_grid(self):
        return self.neon_grid

    def declare_kernel(self, kernel):
        self.kernel = kernel

    def _retrieve_compute_lambda(self):
        return self.kernel

    def _get_tokens(self):
        return self.tokens

    def get_device_id(self):
        return self.gpu_id
