#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:04:45 2017

@author: Allen
"""

# %%

import sys
import os
import csv
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

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


def FormatData(trainX):
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainX /= 255
    print('trainX shape:', trainX.shape)
    print(trainX.shape[0], 'train samples')
    
    return trainX, input_shape

def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))
            
class History(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        
# %%

if __name__ == "__main__":

    # data pre-processing
    trainX, trainY = LoadTrainData(sys.argv[2])

    trainX, input_shape = FormatData(trainX, testX)
    
    samples = int(trainX.shape[0] * 0.85)
    x_train = trainX[:samples, :, :, :]
    y_train = trainY[:samples]
    x_vali = trainX[samples:, :, :, :]
    y_vali = trainY[samples:]
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_vali = keras.utils.to_categorical(y_vali, num_classes)
    
# %% traning
    
    epochs = 100
    batch_size = 128
   
    dataGen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(1024, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(2048, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  # optimizer=keras.optimizers.Adam(),
                  # optimizer=keras.optimizers.SGD(lr=0.01, decay=0.0),
                  metrics=['accuracy'])
    
    
# %% fit model

#    if not os.path.isdir(model_path):
#        os.mkdir(model_path)

    history = History()
    tbCallBack = keras.callbacks.TensorBoard(log_dir=model_path+'/tb', histogram_freq=0, write_graph=True, write_images=True)

#    """
    # fit with image generator
    dataGen.fit(x_train)
    model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train)/32,
                        epochs=epochs,
                        validation_data=(x_vali, y_vali),
                        callbacks=[history, tbCallBack])
    
    # save model and history
    # dump_history(model_path, history)
    model.save('model.h5')
    
# %%
    """
    keras.utils.vis_utils.plot_model(model, to_file=os.path.join(model_path, 'keras_model.png'))
    
    score = model.evaluate(x_vali, y_vali, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    """
    
        