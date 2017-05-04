#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:04:45 2017

@author: Allen
"""

import csv
import numpy as np
import itertools
import keras
from keras import backend as K
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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


#%%
    model_path = 'model.h5'
    emotion_classifier = keras.models.load_model(model_path)

    # np.set_printoptions(precision=2)
    predictions = emotion_classifier.predict_classes(x_vali)
    conf_mat = confusion_matrix(y_vali, predictions)

    fig = plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    fig.savefig('confuse.png')
    # plt.show()