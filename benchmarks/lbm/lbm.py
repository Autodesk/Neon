DOMAIN_SIZE_LIST = "64 128 192 256 320 384 448 512".split()
DEVICE_ID_LIST = "0 1 2 3 4 5 6 7".split()
DEVICE_TYPE_LIST = 'cpu gpu'.split()
GRID_LIST = "dGrid bGrid eGrid".split()
STORAGE_FP_LIST = "double float".split()
COMPUTE_FP_LIST = "double float".split()
OCC_LIST = "nOCC sOCC".split()
HU_LIST = "huGrid huLattice".split()
CURVE_LIST = "sweep morton hilbert".split()
COLLISION_LIST = "bgk kbc".split()
LATTICE_LIST = "d3q19 d3q27".split()
STREAMINGMETHOD_LIST = "push pull aa".split()
TRANSFERMODE_LIST = "get put".split()
STENCILSEMANTIC_LIST = "grid, streaming".split()
WARM_UP_ITER = 10
MAX_ITER = 10000
REPETITIONS = 5

import subprocess
import sys


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
    for DEVICE_TYPE in DEVICE_TYPE_LIST:
        DEVICE_SET_LIST = [DEVICE_ID_LIST[0]]
        if DEVICE_TYPE == 'gpu':
            for DEVICE in DEVICE_ID_LIST[1:]:
                DEVICE_SET_LIST.append(DEVICE_SET_LIST[-1] + ' ' + DEVICE)
        for DEVICE_SET in DEVICE_SET_LIST:
            for OCC in OCC_LIST:
                for DOMAIN_SIZE in DOMAIN_SIZE_LIST:
                    for STORAGE_FP in STORAGE_FP_LIST:
                        for COMPUTE_FP in COMPUTE_FP_LIST:
                            for GRID in GRID_LIST:
                                for HU in HU_LIST:
                                    for CURVE in CURVE_LIST:
                                        for LATTICE in LATTICE_LIST:
                                            for TRANSFERMODE in TRANSFERMODE_LIST:
                                                for STENCILSEMANTIC in STENCILSEMANTIC_LIST:
                                                    for COLLISION in COLLISION_LIST:
                                                        if LATTICE != "d3q27" and LATTICE != "D3Q27":
                                                            continue
                                                        for STREAMINGMETHOD in STREAMINGMETHOD_LIST:
                                                            if STREAMINGMETHOD != 'pull' and len(DEVICE_ID_LIST) != 1:
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
    for DEVICE_TYPE in DEVICE_TYPE_LIST:
        DEVICE_SET_LIST = [DEVICE_ID_LIST[0]]
        if DEVICE_TYPE == 'gpu':
            for DEVICE in DEVICE_ID_LIST[1:]:
                DEVICE_SET_LIST.append(DEVICE_SET_LIST[-1] + ' ' + DEVICE)
        for DEVICE_SET in DEVICE_SET_LIST:
            for OCC in OCC_LIST:
                for DOMAIN_SIZE in DOMAIN_SIZE_LIST:
                    for STORAGE_FP in STORAGE_FP_LIST:
                        for COMPUTE_FP in COMPUTE_FP_LIST:
                            for GRID in GRID_LIST:
                                for HU in HU_LIST:
                                    for CURVE in CURVE_LIST:
                                        for LATTICE in LATTICE_LIST:
                                            for TRANSFERMODE in TRANSFERMODE_LIST:
                                                for STENCILSEMANTIC in STENCILSEMANTIC_LIST:
                                                    for COLLISION in COLLISION_LIST:
                                                        if LATTICE != "d3q27" and LATTICE != "D3Q27":
                                                            continue
                                                        for STREAMINGMETHOD in STREAMINGMETHOD_LIST:
                                                            if STREAMINGMETHOD != 'pull' and len(DEVICE_ID_LIST) != 1:
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
                                                            parameters.append('--max-iter ' + str(MAX_ITER))
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
                                                            parameters.append('--warmup-iter ' + str(WARM_UP_ITER))
                                                            parameters.append('--repetitions ' + str(REPETITIONS))

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
