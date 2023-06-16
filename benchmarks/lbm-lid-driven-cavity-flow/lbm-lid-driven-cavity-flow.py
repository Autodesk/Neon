import subprocess

DOMAIN_SIZE_LIST = "64 128 192 256 320 384 448 512".split()
DEVICE_ID_LIST = "0 1 2 3 4 5 6 7".split()
DEVICE_TYPE_LIST = 'cpu gpu'.split()
GRID_LIST = "dGrid bGrid eGrid".split()
STORAGE_FP_LIST = "double float".split()
COMPUTE_FP_LIST = "double float".split()
OCC_LIST = "nOCC".split()
WARM_UP_ITER = 10
MAX_ITER = 100
REPETITIONS = 5

for DEVICE_TYPE in DEVICE_TYPE_LIST:

    DEVICE_SET_LIST = [DEVICE_ID_LIST[0]]
    if DEVICE_TYPE == 'gpu':
        for DEVICE in DEVICE_ID_LIST[1:]:
            DEVICE_SET_LIST.append(DEVICE_SET_LIST[-1] + ' ' + DEVICE)
    for OCC in OCC_LIST:
        for DOMAIN_SIZE in DOMAIN_SIZE_LIST:
            for STORAGE_FP in STORAGE_FP_LIST:
                for COMPUTE_FP in COMPUTE_FP_LIST:
                    for DEVICE_SET in DEVICE_SET_LIST:

                        if STORAGE_FP == 'double' and COMPUTE_FP == 'float':
                            continue

                        command = 'lbm-lid-driven-cavity-flow'
                        parameters = []
                        parameters.append('--deviceType ' + DEVICE_TYPE)
                        parameters.append('--deviceIds ' + DEVICE_SET)
                        parameters.append('--grid ' + DEVICE_TYPE)
                        parameters.append('--domain-size ' + DOMAIN_SIZE)
                        parameters.append('--warmup-iter ' + str(WARM_UP_ITER))
                        parameters.append('--repetitions ' + str(REPETITIONS))
                        parameters.append('--max-iter ' + str(MAX_ITER))
                        parameters.append(
                            '--report-filename ' + 'lbm-lid-driven-cavity-flow___' +
                            DEVICE_TYPE + '_' + DOMAIN_SIZE + '_' +
                            STORAGE_FP + '_' + COMPUTE_FP + '_' +
                            DEVICE_SET.replace(' ', '_') + '_' + OCC)
                        parameters.append('--computeFP ' + COMPUTE_FP)
                        parameters.append('--storageFP ' + STORAGE_FP)
                        parameters.append('--benchmark')
                        parameters.append('--' + OCC)

                        subprocess.run(['echo' , ' '.join(parameters)])
