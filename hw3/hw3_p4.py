#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:04:45 2017

@author: Allen
"""

# %%

import os
import csv
import numpy as np
import keras
from keras import backend as K
from matplotlib import pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency
from PIL import Image

num_classes = 7
img_rows = 48
img_cols = 48

def LoadTrainData(file_name):
    f = open (file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    data = np.array(data)

    trainX = data[1:, 1]
    images = []
    for row in trainX:
        images.append(np.fromstring(row, dtype=int, sep=' '))
    trainX = np.array(images)
        
    trainY = data[1:, 0]
    
    return np.array(trainX, dtype=np.float), np.array(trainY, dtype=np.int)

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

def FormatData(trainX, testX):
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainX /= 255
    testX /= 255
    print('trainX shape:', trainX.shape)
    print(trainX.shape[0], 'train samples')
    print(testX.shape[0], 'test samples')
    
    return trainX, testX, input_shape
        
# %%

if __name__ == "__main__":
    
    model = keras.models.load_model('model.h5')
    
    # data pre-processing
    trainX, trainY = LoadTrainData('train.csv')
    testX = LoadTestData('test.csv')

    trainX, testX, input_shape = FormatData(trainX, testX)
    
    samples = int(trainX.shape[0] * 0.85)
    x_train = trainX[:samples, :, :, :]
    y_train = trainY[:samples]
    x_vali = trainX[samples:, :, :, :]
    y_vali = trainY[samples:]
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_vali = keras.utils.to_categorical(y_vali, num_classes)

    x_test = testX

# %%
    
    heatmaps = []
    for idx in [160, 9712, 124, 1542]:
        
        folder = 'out{}'.format(idx)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        
        x = x_train[idx] * 255
        x = np.array(x, dtype='uint8')
    
        img = Image.fromarray(x.reshape((48, 48)))
        img.save(os.path.join(folder, 'origin.png'))
        
        pred_class = np.argmax(model.predict(x.reshape((1, 48, 48, 1))))
    
    
        i = 12
            
        heatmap = visualize_saliency(model, i, [pred_class], x)
        #heatmap = visualize_saliency(model, i, [j for j in range(0,64)], x)
        heatmaps.append(heatmap)

    plt.axis('off')
    plt.imshow(utils.stitch_images(heatmaps))
    plt.title('Saliency map')
    plt.savefig(os.path.join(folder, 'stitch.png'), dpi=100)
    