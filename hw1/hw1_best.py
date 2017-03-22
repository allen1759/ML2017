#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:49:45 2017

@author: Allen
"""

# %%

import sys
import csv
import numpy as np

HOURS = 24
NUM_OF_ATTR = 18
DAYS_OF_MON = 20
PM_IND = 9
RF_IND = 10
PERIOD = DAYS_OF_MON * HOURS
TEST_DIM = 10

def LoadTrainData(file_name):
    f = open (file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    
    arr = np.array(data)
    arr = arr[1:, 3:]
    return arr

def LoadTestData(file_name):
    f = open(file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()

    arr = np.array(data)
    arr = arr[:, 2:]

    return np.split(arr, range(NUM_OF_ATTR, arr.shape[0], NUM_OF_ATTR), axis=0)

def ConcateArray(arr, rows):    
    l = np.split(arr, range(rows, arr.shape[0], rows), axis=0)
    concat = np.concatenate(l, axis=1)
    
    return concat

def GetFeature(arr):
    return arr[PM_IND:PM_IND+1, :]

# %%

if __name__ == "__main__":

    # data pre-processing
    arr = LoadTrainData(sys.argv[1])
    arr = ConcateArray(arr, NUM_OF_ATTR)
    
    test_list = LoadTestData(sys.argv[2])
    
# %%
    feature = GetFeature(arr)
    feature = np.array(feature, dtype=np.float)
    
    trainX = []
    trainY = []
    
#    """ two pm2.5
    split_feature = np.split(feature, range(PERIOD, feature.shape[1], 24), axis=1)
    for fea in split_feature:
        tmp = fea[:, 0:TEST_DIM-1].flatten()
        trainX.append(np.append(tmp, 1.0))
        trainY.append(fea[0, TEST_DIM-1])
        tmp = fea[:, fea.shape[1]-TEST_DIM:fea.shape[1]-1]
        trainX.append(np.append(tmp, 1.0))
        trainY.append(fea[0, fea.shape[1]-1])
#    """
        
    """    all pm2.5
    split_feature = np.split(feature, range(PERIOD, feature.shape[1], PERIOD), axis=1)
    for fea in split_feature:
        for i in range(0, fea.shape[1]-TEST_DIM+1):
            tmp = fea[:, i:i+TEST_DIM-1].flatten()
            trainX.append(np.append(tmp, 1.0))
            trainY.append(fea[0, i+TEST_DIM-1])
    """    

# %% calculate

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    
    x = np.linalg.inv(trainX.T.dot(trainX)).dot(trainX.T).dot(trainY)
    
# %% write file
    
    out = []
    for test in test_list:
        fea = GetFeature(test)
        tmp = fea.flatten()
        tmp = np.append(tmp, 1.0)
        
        out.append(np.inner(x, np.array(tmp, dtype=np.float64)))
        
    out = [max(0.0, t) for t in out]
    f = open(sys.argv[3], 'w')
    f.write('id,value\n')
    for i in range(0, len(out)):
        f.write('id_' + str(i) + ',' + str(out[i]) + '\n')
    f.close()
