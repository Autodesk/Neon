import copy
import ctypes
from enum import Enum

import nvtx
import typing
import warp as wp

import neon


class Container:
    # define an enum class
    class ContainerRuntime(Enum):
        warp = 1
        neon = 2

    # This is a set of compiled executable modules loaded by Warp.
    # When getting kernel hooks, we can retain the module references here
    # to prevent them from being unloaded prematurely.

    def __init__(self,
                 loading_lambda=None,
                 execution: neon.Execution = neon.Execution.device(),
                 name = "neon_container"):

        if loading_lambda is None:
            raise Exception('Container: Invalid loading lambda')


        self.api_delete = None
        self.loading_lambda = None
        self.grid = None
        self.backend = None

        self.name  = name
        self.execution = execution

        container_parser: neon.Loader = neon.Loader(execution=execution,
                                                    gpu_id=0,
                                                    data_view=neon.DataView.standard(),
                                                    parsing=True
                                                    )


        self.loading_lambda = loading_lambda
        self.loading_lambda(container_parser)
        self.grid = container_parser._retrieve_grid()

        # We can load the C-API only after the grid is set
        grid_name = self.grid.get_name()
        self.help_load_api(grid_name = grid_name)

        self.backend = self.grid.get_backend()
        # Setting up the information of the Neon container for Neon runtime
        n_devices = self.backend.get_num_devices()  # rows
        self.retained_executable_modules = [set() for _ in range(n_devices)]

        n_data_views = 3  # columns
        # Create a NumPy array of object dtype
        self.k_2Darray = (ctypes.c_void_p * (n_data_views * n_devices))()

        for dev_idx in range(n_devices):
            for dw_idx in range(n_data_views):
                if grid_name == 'mGrid' and dw_idx != 0:
                    # For mGrid at the moment we only support the STANDARD data view
                    continue
                dev_kernel = self._get_kernel(execution=execution,
                                              gpu_id=dev_idx,
                                              data_view=neon.DataView.from_int(dw_idx),
                                              container_runtime=Container.ContainerRuntime.neon)
                # using self.k for debugging
                offset = dev_idx * n_data_views + dw_idx
                dev_str = self.backend.get_device_name(dev_idx)
                k_hook = self._get_kernel_hook(dev_kernel, dev_str, dev_idx)

                self.k_2Darray[offset] = k_hook

        # debug = True
        # if debug:
        #     print("k_2Darray")
        #     for i in range(n_devices):
        #         for j in range(n_data_views):
        #             print(f"Device {i}, DataView {j} hook {hex(k_2Darray[i * n_data_views + j])}")

        self.container_handle = self.neon_gate.handle_type(0)
        block_size = neon.Index_3d(128, 0, 0)
        # Search a function in the .so by composing the function name

        if container_parser.mres_level is None:
            self.api_new(ctypes.pointer(self.container_handle),
                         execution,
                         self.backend.cuda_driver_handle,
                         self.grid.get_handle(),
                         self.k_2Darray,
                         block_size)
        else:
            self.api_mres_new(ctypes.pointer(self.container_handle),
                         container_parser.mres_level,
                         execution,
                         self.backend.cuda_driver_handle,
                         self.grid.get_handle(),
                         self.k_2Darray,
                         block_size)

        self._parsing(container_parser)

    def _parsing(self, parser):
        lib_obj = self.neon_gate.lib
        tokens = parser._get_tokens()
        for token in tokens:
            field = token.get_field()
            access = token.get_access()
            operation = token.get_operation()
            discretization = token.get_discretization()

            field_card = field.get_cardinality()
            field_type = field.get_type()
            field_type_name = ''
            try:
                field_type_name = self.neon_gate.warp_type_to_string[field_type]
            except KeyError:
                raise Exception(f'Unsupported field type {field_type}')
            grid_name = field.get_grid().get_name()

            register_token = getattr(lib_obj, f'warp_container_add_parse_token_{grid_name}_{field_type_name}_{0}')
            register_token.argtypes = [self.neon_gate.handle_type,
                                       self.neon_gate.handle_type,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int]
            register_token.restype = ctypes.c_int

            register_token(self.container_handle,
                           field.get_handle(),
                           access.value,
                           operation.value,
                           discretization.value)

        parse = lib_obj.warp_container_parse
        parse.argtypes = [self.neon_gate.handle_type]
        parse.restype = ctypes.c_int

        parse(self.container_handle)

    def _get_kernel_hook(self, kernel, decvice_str, dev_idx):
        """
         decvice_str = "cuda:0"
        :param kernel:
        :param device_str:
        :return:
        """

        device = wp.get_device(decvice_str)
        # compile and load the executable module
        module_exec = kernel.module.load(device)
        if module_exec is None:
            raise RuntimeError(f"Failed to load module for kernel {kernel.key}")
        self.retained_executable_modules[dev_idx].add(module_exec)
        return module_exec.get_kernel_hooks(kernel).forward

    def help_load_api(self, grid_name: str):
        try:
            self.neon_gate: neon.Gate = neon.Gate()
        except Exception as e:
            self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))

        # ------------------------------------------------------------------
        api_gate = self.neon_gate.lib
        if grid_name != "mGrid":
            self.api_new = getattr(self.neon_gate.lib, f'warp_{grid_name}_container_new')
            self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                     neon.Execution,
                                     self.neon_gate.handle_type,
                                     self.neon_gate.handle_type,
                                     ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(neon.Index_3d)]
            self.api_new.restype = ctypes.c_int
        else:
            self.api_mres_new = getattr(self.neon_gate.lib, f'warp_{grid_name}_container_new')
            self.api_mres_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                     ctypes.c_int32, #Level
                                     neon.Execution,
                                     self.neon_gate.handle_type,
                                     self.neon_gate.handle_type,
                                     ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(neon.Index_3d)]
            self.api_mres_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # warp_container_delete
        self.api_delete = api_gate.warp_container_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = None
        # ------------------------------------------------------------------
        # warp_container_run
        self.api_run = api_gate.warp_container_run
        self.api_run.argtypes = [self.neon_gate.handle_type,
                                 ctypes.c_int,
                                 neon.DataView]
        self.api_run.restype = None
        # ------------------------------------------------------------------
        # parse_token

        # TODOMATT get num devices
        # TODOMATT get device type


    def _get_kernel(self,
                    container_runtime: ContainerRuntime,
                    execution: neon.Execution,
                    gpu_id: int,
                    data_view: neon.DataView):
        span = None
        if self.grid.get_name() != "mGrid":
            span = self.grid.get_span(execution=execution,
                                  dev_idx=gpu_id,
                                  data_view=data_view)
        else:
            span = self.grid.get_span( grid_level = 0,
                                       execution=execution,
                                       dev_idx=gpu_id,
                                       data_view=data_view,
                                       )
        loader: neon.Loader = neon.Loader(execution=execution,
                                          gpu_id=gpu_id,
                                          data_view=data_view)

        self.loading_lambda(loader)
        compute_lambda = loader._retrieve_compute_lambda()

        if container_runtime == Container.ContainerRuntime.warp:
            @wp.kernel
            def kernel():
                x, y, z = wp.tid()
                # wp.printf("WARP my kernel - tid: %d %d %d\n", x, y, z)
                myIdx = wp.neon_set(span, x, y, z)
                # print("my kernel - myIdx: ")
                # wp.neon_print(myIdx)
                compute_lambda(myIdx)

            return kernel

        elif container_runtime == Container.ContainerRuntime.neon:
            @wp.kernel
            def kernel():
                is_active = wp.bool(False)
                myIdx = wp.neon_set(span, is_active)
                if is_active:
                    # print("NEON-RUNTIME kernel - myIdx: ")
                    # wp.neon_print(myIdx)
                    compute_lambda(myIdx)

            return kernel

    def _run_warp(
            self,
            stream_idx: int,
            data_view: neon.DataView):
        # Throw exception as this operation is not supported
        raise Exception('Container: Warp runtime is not supported')
        # """
        # Executing a container in the warp backend.
        # :param stream_idx:
        # :param data_view:
        # :return:
        # """
        # nvtx.push_range(f"{self.name}_warp", color="red")
        #
        # bk = self.grid.get_backend()
        # n_devices = bk.get_num_devices()
        # wp_device_name: str = bk.get_warp_device_name()
        #
        # for dev_idx in range(n_devices):
        #     wp_device = f"{wp_device_name}:{dev_idx}"
        #     span = self.grid.get_span(execution=self.execution,
        #                               dev_idx=dev_idx,
        #                               data_view=data_view)
        #     thread_space = span.get_thread_space()
        #     kernel = self._get_kernel(
        #         container_runtime=Container.ContainerRuntime.warp,
        #         execution=self.execution,
        #         gpu_id=dev_idx,
        #         data_view=data_view)
        #
        #     wp_kernel_dim = thread_space.to_wp_kernel_dim()
        #     wp.launch(kernel, dim=wp_kernel_dim, device=wp_device)
        #     # TODO@Max - WARNING - the following synchronization is temporary
        #     wp.synchronize_device(wp_device)
        #
        # nvtx.pop_range()

    def _run_neon(
            self,
            stream_idx: int,
            data_view: neon.DataView):
        nvtx.push_range(f"{self.name}_neon", color="green")
        self.neon_gate.lib.warp_container_run(self.container_handle,
                                              stream_idx,
                                              data_view)
        nvtx.pop_range()

    def run(self,
            stream_idx: int,
            data_view: neon.DataView = neon.DataView.standard(),
            container_runtime: ContainerRuntime = ContainerRuntime.warp):
        if container_runtime == Container.ContainerRuntime.warp:
            self._run_warp(stream_idx=stream_idx,
                           data_view=data_view)
        elif container_runtime == Container.ContainerRuntime.neon:
            self._run_neon(stream_idx=stream_idx,
                           data_view=data_view)

    @staticmethod
    def factory(name=None):
        def factory_decorator(loading_lambda_generator):
            def container_generator(*args, **kwargs):
                loading_lambda = loading_lambda_generator(*args, **kwargs)
                local_name = copy.deepcopy(name)
                if local_name is None:
                    local_name = f"{loading_lambda.__name__}_neon_container"
                container = Container(loading_lambda=loading_lambda, name = local_name)
                return container

            return container_generator
        return factory_decorator

    @staticmethod
    def fill(field: typing.Any,
             fill_value: typing.Any):

        @Container.factory(name="Fill")
        def container_fill(field):
            def fill_container(loader: neon.Loader):
                loader.set_grid(field.get_grid())
                f = loader.get_write_handle(field)

                @wp.func
                def foo(idx: typing.Any):
                    for c in range(wp.neon_cardinality(f)):
                        wp.neon_write(f, idx, c, fill_value)

                loader.declare_kernel(foo)

            return fill_container

        ret = container_fill(field=field)
        return ret

    @staticmethod
    def zero(field):
        ret = Container.fill(field=field, fill_value=field.type(0))
        return ret

    @staticmethod
    def copy(field_src: typing.Any,
             field_dst: typing.Any):

        @Container.factory(name="copy")
        def copy_container(field_src, field_dst):
            def copy_ll(loader: neon.Loader):
                loader.set_grid(field_src.get_grid())
                src_pn = loader.get_read_handle(field_src)
                dst_pn = loader.get_read_handle(field_dst)

                @wp.func
                def copy_cl(idx: typing.Any):
                    for c in range(wp.neon_cardinality(src_pn)):
                        val = wp.neon_read(src_pn, idx, c)
                        wp.neon_write(dst_pn, idx, c, val)

                loader.declare_kernel(copy_cl)
            return copy_ll

        ret = copy_container(field_src=field_src, field_dst=field_dst)
        return ret