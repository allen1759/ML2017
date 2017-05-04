#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:04:45 2017

@author: Allen
"""

# %%

import sys
import csv
import numpy as np
import keras
from keras import backend as K

num_classes = 7
img_rows = 48
img_cols = 48

def LoadTestData(file_name):
    f = open (file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    data = np.array(data)

    testX = data[1:, 1]
    images = []
    for row in testX:
        images.append(row.split(' '))
    testX = np.array(images)
        
    return np.array(testX, dtype=np.float)

def FormatData(testX):
    if K.image_data_format() == 'channels_first':
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    testX /= 255
    print(testX.shape[0], 'test samples')
    
    return testX
            
def WriteResult(result, file_name):
    f = open(file_name, 'w')
    f.write('id,label\n')
    for i in range(0, len(result)):
        f.write(str(i) + ',' + str(result[i]) + '\n')
    f.close()
        
# %%

if __name__ == "__main__":

    testX = LoadTestData(sys.argv[1])
    testX = FormatData(testX)
    
    x_test = testX
    
# %% load model from file

    epochs = 50
    batch_size = 128
    
    model = keras.models.load_model('model.h5')   
    
# %% write file

    result = model.predict_classes(x_test, batch_size=batch_size)

    WriteResult(result, sys.argv[2])
        