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
import matplotlib.pyplot as plt

# %%
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
    
    filter_dir = 'filterOut2'
    
    emotion_classifier = keras.models.load_model('model8.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    
# %%
    input_img = emotion_classifier.input
    name_ls = ['conv2d_42', 'conv2d_43']
    #name_ls = ['conv2d_10', 'conv2d_11']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


    choose_id = 17
    x_vali = [e.reshape((1, 48, 48, 1)) for e in x_vali]
    photo = x_vali[choose_id]
    
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))

        if not os.path.isdir(filter_dir):
            os.mkdir(filter_dir)
        fig.savefig(os.path.join(filter_dir,'layer{}'.format(cnt)))