#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:04:45 2017

@author: Allen
"""

# %%

import sys
import csv
import random
import numpy as np

def LoadTrainData(file_nameX, file_nameY):
    f = open (file_nameX, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    trainX = np.array(data)
    
    f = open (file_nameY, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    
    trainY = np.array(data, dtype=np.int)
    return np.array(trainX[1:, :], dtype=np.float), trainY.flatten()

def SplitClasses(trainX, trainY):
    class1 = []
    class2 = []
    for row1, row2 in zip(trainX, trainY):
        if row2 == 1:
            class1.append(row1)
        else:
            class2.append(row1)
            
    return np.array(class1), np.array(class2)

def LoadTestData(file_name):
    f = open(file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()

    testX = np.array(data)
    return np.array(testX[1:, :], dtype=np.float)

def Normalize(trainX, testX):
    return trainX / trainX.max(axis=0), testX / trainX.max(axis=0)

def Sigmoid(z):
    zpro = [max(-100, i) for i in z]
    zpro = np.array(zpro)
    return 1 / (1 + np.exp(-zpro))

def Gradient(trainX, trainY, iterative, lr):
    weight = [random.uniform(-1, 1) for i in range(trainX.shape[1])]
    weight = np.array(weight)
    bias = [random.uniform(-1, 1)]
    bias = np.array(bias)
    trainY = np.array(trainY, dtype=np.float).flatten()

    adasum_w = 1e-10
    adasum_b = 1e-10
    for i in range(0, iterative):
        y = np.dot(trainX, weight) + bias
        L = Sigmoid(y) - trainY
        gra_w = np.dot(trainX.T, L)
        gra_b = np.sum(L)
        adasum_w += gra_w ** 2
        adasum_b += gra_b ** 2
        ada_w = np.sqrt(adasum_w)
        ada_b = np.sqrt(adasum_b)
        weight -= lr * gra_w / ada_w
        bias -= lr * gra_b / ada_b
        
    return weight, bias

def WriteResult(w, b, testX, file_name):
    out = []
    for row in testX:
        x = np.matrix(row)
        if np.inner(x, w) + b > 0.5:
            out.append(1)
        else:
            out.append(0)
    
    f = open(file_name, 'w')
    f.write('id,label\n')
    for i in range(0, len(out)):
        f.write(str(i+1) + ',' + str(out[i]) + '\n')
    f.close()

# %%

if __name__ == "__main__":

    # data pre-processing
    trainX, trainY = LoadTrainData(sys.argv[1], sys.argv[2])
    testX = LoadTestData(sys.argv[3])
    class1, class2 = SplitClasses(trainX, trainY)
    
    trainX, testX = Normalize(trainX, testX)
    
# %% traning
    
    trainX2 = trainX[:, :6] ** 2
    testX2 = testX[:, :6] ** 2
    
    for i in range(6):
        for j in range(i+1, 6):
            trainX2 = np.append(trainX2, trainX[:, i:i+1]*trainX[:, j:j+1], axis=1)
            testX2 = np.append(testX2, testX[:, i:i+1]*testX[:, j:j+1], axis=1)
            
    
    trainX2 = np.append(trainX2, trainX, axis=1)
    testX2 = np.append(testX2, testX, axis=1)
    
#    w, b = Gradient(trainX2, trainY, 20000, 1.0)
    
# %% load model

    with open('model', 'r') as f:
        line = f.readline()
        w = np.array(line.split(' '), dtype=np.float)
        line = f.readline()
        b = np.array([line], dtype=np.float)

# %% write file

    WriteResult(w, b, testX2, sys.argv[4])
