deviceType_LIST = 'gpu'.split()
deviceIds_LIST = "0 1 2 3 4 5 6 7".split()
grid_LIST = "dGrid dGridDisg".split()
domainSize_LIST = "352 368 384 400 416 432 448 464 480 496 512".split()
computeFP_LIST = "double".split()
storageFP_LIST = "double".split()
occ_LIST = "standard".split()
transferMode_LIST = "get".split()
stencilSemantic_LIST = "lattice".split()
spaceCurve_LIST = "sweep".split()
collision_LIST = "bgk".split()
streamingMethod_LIST = "pull".split()
lattice_LIST = "d3q19".split()

warmupIter_INT = 10
repetitions_INT = 1
maxIter_INT = 10000

execute_for_efficiency_max_num_devices = False
execute_by_skipping_single_gpu = False
execute_single_gpu_only = False

# deviceType_LIST = 'gpu'.split()
# deviceIds_LIST = "0 1 2 3 4 5 6 7".split()
# grid_LIST = "dGrid bGridMgpu_4_4_4".split()
# domainSize_LIST = "64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512".split()
# computeFP_LIST = "double float".split()
# storageFP_LIST = "double float".split()
# occ_LIST = "none standard".split()
# transferMode_LIST = "get put".split()
# stencilSemantic_LIST = "standard lattice".split()
# spaceCurve_LIST = "sweep morton hilbert".split()
# collision_LIST = "bgk kbc".split()
# streamingMethod_LIST = "push pull aa".split()
# lattice_LIST = "d3q19 d3q27".split()
#
# warmupIter_INT = 10
# repetitions_INT = 5
# maxIter_INT = 10000
#
# goal_is_efficiency_max_num_devices = False


import subprocess
import sys


class Config_space:
    def __init__(self):
        self.space = {}
        self.count_options = 1

    def add(self, list_):
        # get name of the list in the calling frame
        import inspect
        frame = inspect.currentframe().f_back
        variables = {id(v): k for k, v in frame.f_locals.items()}
        global_list_name = variables.get(id(list_), None)
        assert global_list_name is not None
        # remove suffix _LIST from the global_list_name
        global_list_name = global_list_name[:-5]
        self.space[global_list_name] = list_
        self.count_options *= len(list_)

    def generate_options(self):
        self.configurations = [{}]
        for conf_name in self.space.keys():
            configurations_next = []

            for partial_conf in self.configurations:
                if conf_name == 'deviceIds':
                    new_conf = partial_conf.copy()
                    new_conf[conf_name] = self.space[conf_name]
                    configurations_next.append(new_conf)
                    continue
                for option in self.space[conf_name]:
                    new_conf = partial_conf.copy()
                    new_conf[conf_name] = option
                    configurations_next.append(new_conf)
            self.configurations = configurations_next.copy()
        self.__filtering()

    def get_benchmarks_list(self):
        return self.configurations

    def get_benchmarks_num(self):
        counter = 0
        for conf in self.get_benchmarks_list():
            counter += len(self.expand_device_sets(conf))
        return counter

    def __filtering(self):
        new_configurations = []
        for conf in self.configurations:
            if conf['collision'] == 'kbc' and conf['lattice'] != 'd3q27':
                continue
            if conf['streamingMethod'] != 'pull' and len(conf['deviceIds']) != 1:
                continue
            if conf['storageFP'] == 'double' and conf['computeFP'] == 'float':
                continue
            if conf['storageFP'] == 'float' and conf['computeFP'] == 'double':
                continue
            new_configurations.append(conf)

    def __expand_device_sets(self):
        """
        This function expands the device sets in the configurations.
        The behaviour of this function is controlled by the following global variables:

        :return:
        """
        new_configurations = []
        for conf in self.configurations:
            deviceType = conf['deviceType']
            deviceIds = conf['deviceIds']
            dev_set_list = self.__get_device_set_list(deviceType, deviceIds)
            for dev_set in dev_set_list:
                new_conf = conf.copy()
                new_conf['dev_set_ids'] = dev_set
                new_configurations.append(new_conf)
        self.configurations = new_configurations

    def expand_device_sets(self, conf):
        """
        This function expands the device sets in the configurations.
        The behaviour of this function is controlled by the following global variables:

        :return:
        """
        deviceType = conf['deviceType']
        deviceIds = conf['deviceIds']
        generated_configurations = []
        dev_set_list = self.__get_device_set_list(deviceType, deviceIds)
        for dev_set in dev_set_list:
            new_conf = conf.copy()
            new_conf['dev_set_ids'] = dev_set
            generated_configurations.append(new_conf)
        return generated_configurations

    def __get_device_set_list(self, DEVICE_TYPE, deviceIds_LIST):
        """
        This function returns a list of device configurations.
        The returned list is a list of strings, where each string is a space-separated list of device ids.

        For example: ['0', '0 1', '0 1 2', '0 1 2 3', '0 1 2 3 4', '0 1 2 3 4 5', '0 1 2 3 4 5 6', '0 1 2 3 4 5 6 7']

        The behaviour of this function is controlled by the following global variables:
        - execute_for_efficiency_max_num_devices
        - execute_by_skipping_single_gpu
        - execute_single_gpu

        :param DEVICE_TYPE:
        :param deviceIds_LIST:
        :return:
        """
        assert DEVICE_TYPE in ['gpu', 'cpu']
        assert len(deviceIds_LIST) > 0

        if len(deviceIds_LIST) == 1 or DEVICE_TYPE == 'cpu':
            return [deviceIds_LIST[0]]
        if execute_single_gpu_only:
            return [deviceIds_LIST[0]]
        if execute_for_efficiency_max_num_devices:
            return [deviceIds_LIST[0], ' '.join(deviceIds_LIST)]
        if not execute_for_efficiency_max_num_devices:
            DEVICE_SET_LIST = [deviceIds_LIST[0]]
            offset = 1
            if execute_by_skipping_single_gpu:
                DEVICE_SET_LIST = [deviceIds_LIST[0] + ' ' + deviceIds_LIST[1]]
                offset = 2
            for DEVICE in deviceIds_LIST[offset:]:
                # take a copy of the last element in the list and append the new device
                # add the combine sequence of ids to the list
                DEVICE_SET_LIST.append(DEVICE_SET_LIST[-1] + ' ' + DEVICE)
            return DEVICE_SET_LIST


def printProgressBar(value, label, id, max_id):
    n_bar = 40  # size of progress bar
    max = 100
    j = value / max
    sys.stdout.write('\r')
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))

    sys.stdout.write(f"{label.ljust(7)} [{id} of {max_id}] {(str(int(100 * j)) + '%').rjust(4)} [{bar:{n_bar}s}]    \n")
    sys.stdout.flush()


config_space = Config_space()
config_space.add(deviceType_LIST)
config_space.add(deviceIds_LIST)
config_space.add(grid_LIST)
config_space.add(domainSize_LIST)
config_space.add(computeFP_LIST)
config_space.add(storageFP_LIST)
config_space.add(occ_LIST)
config_space.add(transferMode_LIST)
config_space.add(stencilSemantic_LIST)
config_space.add(spaceCurve_LIST)
config_space.add(collision_LIST)
config_space.add(streamingMethod_LIST)
config_space.add(lattice_LIST)

counter = 0
command = './lbm'
# command = 'echo'


with (open(command + '.log', 'w') as fp):
    config_space.generate_options()
    benchmarks_num = config_space.get_benchmarks_num()
    benchmarks_compressed = []
    benchmarks_expanded = []


    def reset_checkpoint():
        # remove checkpoint files
        import os
        os.remove('checkpoint.txt')


    def store_checkpoint(compressed, expanded, counter_, benchmarks_num_):
        status = {'compressed': compressed,
                  'expanded': expanded,
                  'counter': counter_,
                  'benchmarks_num': benchmarks_num_}

        with open('checkpoint.txt', 'w') as f:
            f.write(str(status))


    def load_checkpoint():
        with open('checkpoint.txt', 'r') as f:
            status = eval(f.read())
            compressed = status['compressed']
            expanded = status['expanded']
            counter_ = status['counter']
            benchmarks_num_ = status['benchmarks_num']
        return compressed, expanded, counter_, benchmarks_num_


    # if there is a checkpoint file, read it and skip the benchmarks that have already been executed
    try:
        benchmarks_compressed, _, _, benchmarks_num = load_checkpoint()
    except Exception as e:
        benchmarks_compressed = config_space.get_benchmarks_list()
        store_checkpoint(benchmarks_compressed, None, counter, benchmarks_num)

    while len(benchmarks_compressed) > 0:
        benchmark_compressed = benchmarks_compressed[0]

        # read the decompressed benchmarks from a checkpoint file it exists
        try:
            _, benchmarks_expanded, counter, _ = load_checkpoint()
            if benchmarks_expanded is None:
                raise Exception('No checkpoint data for expanded benchmarks')
        except Exception as e:
            benchmarks_expanded = config_space.expand_device_sets(benchmark_compressed)
            store_checkpoint(benchmark_compressed, benchmarks_expanded, counter, None)

        while len(benchmarks_expanded) > 0:
            benchmark = benchmarks_expanded[0]

            parameters = []
            parameters.append('--deviceType ' + benchmark['deviceType'])
            parameters.append('--deviceIds ' + benchmark['dev_set_ids'])
            parameters.append('--grid ' + benchmark['grid'])
            parameters.append('--domain-size ' + benchmark['domainSize'])
            parameters.append('--max-iter ' + str(maxIter_INT))
            parameters.append('--report-filename ' + 'lbm')
            parameters.append('--computeFP ' + benchmark['computeFP'])
            parameters.append('--storageFP ' + benchmark['storageFP'])
            parameters.append('--occ ' + benchmark['occ'])
            parameters.append('--transferMode ' + benchmark['transferMode'])
            parameters.append('--stencilSemantic ' + benchmark['stencilSemantic'])
            parameters.append('--spaceCurve ' + benchmark['spaceCurve'])
            parameters.append('--collision ' + benchmark['collision'])
            parameters.append('--streamingMethod ' + benchmark['streamingMethod'])
            parameters.append('--lattice ' + benchmark['lattice'])
            parameters.append('--benchmark ')
            parameters.append('--warmup-iter ' + str(warmupIter_INT))
            parameters.append('--repetitions ' + str(repetitions_INT))

            commandList = []
            commandList.append(command)
            for el in parameters:
                for s in el.split():
                    commandList.append(s)

            fp.write("\n-------------------------------------------\n")
            fp.write(' '.join(commandList))
            fp.write("\n-------------------------------------------\n")
            fp.flush()
            print(' '.join(commandList))
            subprocess.run(commandList, text=True, stdout=fp)

            # if counter == 5:
            #     exit(99)

            counter += 1
            printProgressBar(counter * 100.0 / benchmarks_num, 'Progress', id=counter, max_id=benchmarks_num)

            # checkpoint the decompressed benchmarks
            benchmarks_expanded.pop(0)
            store_checkpoint(benchmarks_compressed, benchmarks_expanded, counter, benchmarks_num)

        benchmarks_compressed.pop(0)
        # checkpointing : remove the decompressed benchmarks and update the compressed benchmarks file
        store_checkpoint(benchmarks_compressed, None, counter, benchmarks_num)

    # remove the compressed benchmarks file
    reset_checkpoint()
    print('All benchmarks have been executed successfully.')