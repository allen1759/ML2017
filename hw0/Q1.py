import sys
import numpy as np

matrixA = []
matrixB = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.split(',')
        if line:
            line = [int(i) for i in line]
            matrixA.append(line)

with open(sys.argv[2]) as f:
    for line in f:
        line = line.split(',')
        if line:
            line = [int(i) for i in line]
            matrixB.append(line)

result = np.dot(matrixA, matrixB)
with open('ans_one.txt', 'w') as f:
    for e in np.sort(result.flatten()):
        f.write(str(e) + '\n')
