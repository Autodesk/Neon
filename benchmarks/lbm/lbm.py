deviceType_LIST = 'gpu'.split()
deviceIds_LIST = "0 1 2 3 4 5 6 7".split()
grid_LIST = "dGrid dGridDisg".split()
domainSize_LIST = "64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512".split()
computeFP_LIST = "double float".split()
storageFP_LIST = "double float".split()
occ_LIST = "standard".split()
transferMode_LIST = "get".split()
stencilSemantic_LIST = "lattice".split()
spaceCurve_LIST = "sweep".split()
collision_LIST = "bgk kbc".split()
streamingMethod_LIST = "pull".split()
lattice_LIST = "d3q19 d3q27".split()

warmupIter_INT = 10
repetitions_INT = 5
maxIter_INT = 10000

goal_is_efficiency_max_num_devices = False

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


# from typing import List, Dict

def getDeviceConfigurations(DEVICE_TYPE, deviceIds_LIST):
    if not goal_is_efficiency_max_num_devices:
        DEVICE_SET_LIST = [deviceIds_LIST[0]]
        if DEVICE_TYPE == 'gpu':
            for DEVICE in deviceIds_LIST[1:]:
                DEVICE_SET_LIST.append(DEVICE_SET_LIST[-1] + ' ' + DEVICE)
        return DEVICE_SET_LIST
    if len(deviceIds_LIST) ==1:
        return  [deviceIds_LIST[0]]

    if  goal_is_efficiency_max_num_devices:
        return [deviceIds_LIST[0], ' '.join(deviceIds_LIST)]


def printProgressBar(value, label):
    n_bar = 40  # size of progress bar
    max = 100
    j = value / max
    sys.stdout.write('\r')
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))

    sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()


def countAll():
    counter = 0
    for DEVICE_TYPE in deviceType_LIST:
        DEVICE_SET_LIST = getDeviceConfigurations(DEVICE_TYPE, deviceIds_LIST)
        for DEVICE_SET in DEVICE_SET_LIST:
            for OCC in occ_LIST:
                for DOMAIN_SIZE in domainSize_LIST:
                    for STORAGE_FP in storageFP_LIST:
                        for COMPUTE_FP in computeFP_LIST:
                            for GRID in grid_LIST:
                                for CURVE in spaceCurve_LIST:
                                    for LATTICE in lattice_LIST:
                                        for TRANSFERMODE in transferMode_LIST:
                                            for STENCILSEMANTIC in stencilSemantic_LIST:
                                                for COLLISION in collision_LIST:
                                                    if LATTICE != "d3q27" and COLLISION == 'kbc':
                                                        continue
                                                    for STREAMINGMETHOD in streamingMethod_LIST:
                                                        if STREAMINGMETHOD != 'pull' and len(DEVICE_SET_LIST) != 1:
                                                            continue
                                                        if STORAGE_FP == 'double' and COMPUTE_FP == 'float':
                                                            continue
                                                        if STORAGE_FP == 'float' and COMPUTE_FP == 'double':
                                                            continue

                                                        counter += 1
    return counter


SAMPLES = countAll()
counter = 0
command = './lbm'
# command = 'echo'
with open(command + '.log', 'w') as fp:
    for DEVICE_TYPE in deviceType_LIST:
        DEVICE_SET_LIST = getDeviceConfigurations(DEVICE_TYPE, deviceIds_LIST)
        for DEVICE_SET in DEVICE_SET_LIST:
            for OCC in occ_LIST:
                for DOMAIN_SIZE in domainSize_LIST:
                    for STORAGE_FP in storageFP_LIST:
                        for COMPUTE_FP in computeFP_LIST:
                            for GRID in grid_LIST:
                                for CURVE in spaceCurve_LIST:
                                    for LATTICE in lattice_LIST:
                                        for TRANSFERMODE in transferMode_LIST:
                                            for STENCILSEMANTIC in stencilSemantic_LIST:
                                                for COLLISION in collision_LIST:
                                                    if LATTICE != "d3q27" and COLLISION == 'kbc':
                                                        continue
                                                    for STREAMINGMETHOD in streamingMethod_LIST:
                                                        if STREAMINGMETHOD != 'pull' and len(DEVICE_SET_LIST) != 1:
                                                            continue
                                                        if STORAGE_FP == 'double' and COMPUTE_FP == 'float':
                                                            continue
                                                        if STORAGE_FP == 'float' and COMPUTE_FP == 'double':
                                                            continue

                                                        parameters = []
                                                        parameters.append('--deviceType ' + DEVICE_TYPE)
                                                        parameters.append('--deviceIds ' + DEVICE_SET)
                                                        parameters.append('--grid ' + GRID)
                                                        parameters.append('--domain-size ' + DOMAIN_SIZE)
                                                        parameters.append('--max-iter ' + str(maxIter_INT))
                                                        parameters.append('--report-filename ' + 'lbm')
                                                        parameters.append('--computeFP ' + COMPUTE_FP)
                                                        parameters.append('--storageFP ' + STORAGE_FP)
                                                        parameters.append('--occ ' + OCC)
                                                        parameters.append('--transferMode ' + TRANSFERMODE)
                                                        parameters.append('--stencilSemantic ' + STENCILSEMANTIC)
                                                        parameters.append('--spaceCurve ' + CURVE)
                                                        parameters.append('--collision ' + COLLISION)
                                                        parameters.append('--streamingMethod ' + STREAMINGMETHOD)
                                                        parameters.append('--lattice ' + LATTICE)
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

                                                        counter += 1
                                                        printProgressBar(counter * 100.0 / SAMPLES, 'Progress')
