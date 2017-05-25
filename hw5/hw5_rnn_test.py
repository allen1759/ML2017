# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:04:45 2017

@author: Allen
"""

import sys
import pickle
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

test_path = sys.argv[1]
output_path= sys.argv[2]
"""
test_path = 'test_data.csv'
output_path = 'output_rnn_test.csv'
"""
#####################
###   parameter   ###
#####################
embedding_dim = 100

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding='utf-8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


# %%
#########################
###   Main function   ###
#########################
if __name__=='__main__':

    ### read training and testing data
    (_, X_test,_) = read_data(test_path,False)
    
#%%
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    word_index = tokenizer.word_index

    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    #train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    #train_sequences = pad_sequences(train_sequences)
    max_article_length = 306
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)


    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        embedding_dim,
                        #weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    model.add(GRU(256,activation='tanh',dropout=0.25))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()
    
    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])

# %%

    tag_list = [ 'SCIENCE-FICTION',
                 'SPECULATIVE-FICTION',
                 'FICTION',
                 'NOVEL',
                 'FANTASY',
                 "CHILDREN'S-LITERATURE",
                 'HUMOUR',
                 'SATIRE',
                 'HISTORICAL-FICTION',
                 'HISTORY',
                 'MYSTERY',
                 'SUSPENSE',
                 'ADVENTURE-NOVEL',
                 'SPY-FICTION',
                 'AUTOBIOGRAPHY',
                 'HORROR',
                 'THRILLER',
                 'ROMANCE-NOVEL',
                 'COMEDY',
                 'NOVELLA',
                 'WAR-NOVEL',
                 'DYSTOPIA',
                 'COMIC-NOVEL',
                 'DETECTIVE-FICTION',
                 'HISTORICAL-NOVEL',
                 'BIOGRAPHY',
                 'MEMOIR',
                 'NON-FICTION',
                 'CRIME-FICTION',
                 'AUTOBIOGRAPHICAL-NOVEL',
                 'ALTERNATE-HISTORY',
                 'TECHNO-THRILLER',
                 'UTOPIAN-AND-DYSTOPIAN-FICTION',
                 'YOUNG-ADULT-LITERATURE',
                 'SHORT-STORY',
                 'GOTHIC-FICTION',
                 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION',
                 'HIGH-FANTASY']

    model.load_weights('best_50471_5751.hdf5')
    
    Y_pred = model.predict(test_sequences)
    
    thresh = 0.4
    with open(output_path,'w',encoding='utf-8') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

