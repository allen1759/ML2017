#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:04:45 2017

@author: Allen
"""

# %%

import sys
import csv
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

def ClacMuSigma(trainX, class1, class2):
    dim = trainX.shape[1]
    trainX = np.matrix(trainX)
    class1 = np.matrix(class1)
    class2 = np.matrix(class2)
    
    mu1 = np.average(class1, axis=0)
    mu2 = np.average(class2, axis=0)
    sigma1 = np.zeros((dim, dim), dtype=np.float)
    for row in class1:
        sigma1 += np.dot((row-mu1).T, (row-mu1))
    sigma1 /= class1.shape[0]
    
    sigma2 = np.zeros((dim, dim), dtype=np.float)
    for row in class2:
        sigma2 += np.dot((row-mu2).T, (row-mu2))
    sigma2 /= class2.shape[0]
    
    sigma = (sigma1 * class1.shape[0] + sigma2 * class2.shape[0]) / trainX.shape[0]
    
    mu1 = np.matrix(mu1)
    mu2 = np.matrix(mu2)
    sigma = np.matrix(sigma)
    sigmai = np.linalg.inv(sigma)
    
    return mu1, mu2, sigma, sigmai

def CalcWB(mu1, mu2, sigmai, num1, num2):
    w = (mu1 - mu2).dot(sigmai)
    b = -0.5 * mu1.dot(sigmai).dot(mu1.T) + 0.5 * mu2.dot(sigmai).dot(mu2.T) + np.log(np.float(num1)/num2)
    return w, b

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
    
# %% training

    mu1, mu2, sigma, sigmai = ClacMuSigma(trainX, class1, class2)
    w, b = CalcWB(mu1, mu2, sigmai, class1.shape[0], class2.shape[0])
    
# %% write file
        
    WriteResult(w, b, testX, sys.argv[4])