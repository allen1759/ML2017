#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:44:35 2017

@author: Allen
"""

import sys
import numpy as np
import pandas as pd
import random
import keras.backend as K
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint


K_FACTORS = 200

class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        
        P_bias = Sequential()
        P_bias.add(Embedding(n_users, 1, input_length=1))
        P_bias.add(Reshape((1,)))
        Q_bias = Sequential()
        Q_bias.add(Embedding(m_items, 1, input_length=1))
        Q_bias.add(Reshape((1,)))
        
        super(CFModel, self).__init__(**kwargs)
        #self.add(Merge([P, Q], mode='dot', dot_axes=1))
        self.add(Merge([Merge([P, Q], mode='dot', dot_axes=1), P_bias, Q_bias], mode='sum'))

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def WriteResult(result, file_name):
    f = open(file_name, 'w')
    f.write('TestDataID,Rating\n')
    for i in range(0, len(result)):
        num = result[i]
        if num<0:
            num = 0.0
        if num>5:
            num = 5.0
        f.write(str(i+1) + ',' + str(num) + '\n')
    f.close()
    
# %%

max_user_id = 6040
max_movie_id = 3952

test = pd.read_csv(sys.argv[1] + 'test.csv',
                   sep=',',
                   usecols=['TestDataID', 'UserID', 'MovieID'])

trained_model = CFModel(max_user_id, max_movie_id, K_FACTORS)
trained_model.load_weights('model_34.h5')

# %%

result = trained_model.predict([test['UserID'].as_matrix()-1, test['MovieID'].as_matrix()-1, np.ones(test['UserID'].as_matrix().shape[0]), np.ones(test['UserID'].as_matrix().shape[0])])
WriteResult(result.flatten(), sys.argv[2])

