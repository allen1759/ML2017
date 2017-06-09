#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:44:35 2017

@author: Allen
"""

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


class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.25, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
        self.add(Dropout(p_dropout))
        self.add(Dense(k_factors, activation='relu'))
        self.add(Dropout(p_dropout))
        self.add(Dense(1, activation='linear'))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def predict_rating(userid, movieid):
    return trained_model.rate(userid, movieid)

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

rating = pd.read_csv('train.csv',
                     sep=',',
                     usecols=['TrainDataID', 'UserID', 'MovieID', 'Rating'])
max_user_id = rating['UserID'].drop_duplicates().max()
max_movie_id = rating['MovieID'].drop_duplicates().max()
print(len(rating), 'ratings loaded.')

RNG_SEED = random.randint(0,100)
shuffled_ratings = rating.sample(frac=1., random_state=RNG_SEED)
movie_emb_id = shuffled_ratings['MovieID'].values - 1
user_emb_id = shuffled_ratings['UserID'].values - 1
rating_value = shuffled_ratings['Rating'].values

# %%

model = CFModel(max_user_id, max_movie_id, K_FACTORS)
#model = DeepModel(max_user_id, max_movie_id, K_FACTORS)
model.compile(loss='mse', optimizer='adam', metrics=[rmse])

# %%

callbacks = [EarlyStopping('val_rmse', patience=10, mode='min'),
             ModelCheckpoint('model.h5', save_best_only=True, mode='min', monitor='val_rmse', verbose=1)]

#history = model.fit([user_emb_id, movie_emb_id], rating_value, epochs=3000, validation_split=0.1, callbacks=callbacks, batch_size=1000)
history = model.fit([user_emb_id, movie_emb_id, np.ones(user_emb_id.shape[0]), np.ones(user_emb_id.shape[0])], rating_value, epochs=30, validation_split=0.1, callbacks=callbacks, batch_size=1000)

# %%

max_user_id = 6040
max_movie_id = 3952

test = pd.read_csv('test.csv',
                   sep=',',
                   usecols=['TestDataID', 'UserID', 'MovieID'])

trained_model = CFModel(max_user_id, max_movie_id, K_FACTORS)
#trained_model = DeepModel(max_user_id, max_movie_id, K_FACTORS)
trained_model.load_weights('model.h5')

# %%

#result = trained_model.predict([test['UserID'].as_matrix()-1, test['MovieID'].as_matrix()-1])
result = trained_model.predict([test['UserID'].as_matrix()-1, test['MovieID'].as_matrix()-1, np.ones(test['UserID'].as_matrix().shape[0]), np.ones(test['UserID'].as_matrix().shape[0])])
WriteResult(result.flatten(), 'submission.csv')

    # %%
