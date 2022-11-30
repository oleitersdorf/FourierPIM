
import os
import subprocess
import select


def runTest(file, vectorSize, batchSize, numIterations):

    process = subprocess.Popen(['./' + file, str(vectorSize), str(batchSize), str(numIterations)], stdout=subprocess.PIPE)
    p = select.poll()
    p.register(process.stdout, select.POLLIN)

    start = process.stdout.readline().decode('ascii')
    assert(str(start) == "START\n")

    totalPower = 0
    totalMemory = 0
    numMeasurements = 0
    while True:

        # Poll for power and memory
        m = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw,memory.used', '--format=csv,noheader']).decode('ascii').split(' ')
        totalPower += float(m[0])
        totalMemory += float(m[2])
        numMeasurements += 1

        # Check if program finished
        if p.poll(1):
            stop = process.stdout.readline().decode('ascii')
            assert(str(stop) == "STOP\n")
            break

    time = float(process.stdout.readline().decode('ascii'))/1000
    power = totalPower/numMeasurements
    memory = totalMemory/numMeasurements

    print(f'{file}({vectorSize}, {batchSize}): {round(time*1000, 1)} ms, {round(power, 1)} W, {round(memory, 1)} MiB -> {round(batchSize/time, 1)} Tput, {round(batchSize/(time*power), 1)} Tput/W')


def runTests(file, vectorSizes, batchSizes, numIterations):
    for i in range(len(vectorSizes)):
        runTest(file, vectorSizes[i], batchSizes[i], numIterations)


def runAllTests():

    # Compile for full-precision
    os.system('make -s DTYPE=full')
    # Run full-precision tests
    print('Full Precision:')
    runTests('fft', [1024, 2048, 4096, 8192], [500000, 250000, 125000, 62500], 1024)
    runTests('complex_poly', [1024, 2048, 4096], [330000, 165000, 82500], 1024)
    runTests('real_poly', [1024, 2048, 4096, 8192], [285000, 142500, 71250, 35625], 1024)
    print()

    # Compile for half-precision
    os.system('make -s DTYPE=half')
    # Run half-precision tests
    print('Half Precision:')
    runTests('fft', [1024, 2048, 4096, 8192, 16384], [650000, 325000, 162500, 81250, 40625], 1024)
    runTests('complex_poly', [1024, 2048, 4096, 8192], [500000, 250000, 125000, 62500], 1024)
    runTests('real_poly', [1024, 2048, 4096, 8192, 16384], [400000, 200000, 100000, 50000, 25000], 1024)


if __name__ == '__main__':
    runAllTests()
