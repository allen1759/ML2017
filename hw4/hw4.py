# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:54:47 2017

@author: Allen
"""

import sys
import numpy as np
from sklearn import cluster

def WriteResult(result, file_name):
    f = open(file_name, 'w')
    f.write('SetId,LogDim\n')
    for i in range(0, len(result)):
        f.write(str(i) + ',' + str(np.log(result[i])) + '\n')
    f.close()

# %%

data = np.load(sys.argv[1])

# %%

varis = []
for i in range(200):
    x = data[str(i)]
    varis.append(np.var(x))

x_train = np.array(varis)

model = cluster.KMeans(init='k-means++', n_clusters=60, random_state=42)
kmeans = model.fit(x_train.reshape((x_train.shape[0], 1)))

centers = np.array(kmeans.cluster_centers_)
centers = np.sort(centers.flatten())

# %%

result = []

for x in x_train:
    ind = np.interp(x, centers, range(60))
    if ind < 0:
        ind = 0
    elif ind > 59:
        ind = 59
    result.append(ind + 1)

result = np.array(result, dtype=np.float32)
        
# %%

WriteResult(result, sys.argv[2])

# %%

"""
# predict hand rotate
import os
from PIL import Image

var = []
for i in range(481):
    img = Image.open(os.path.join('hand', 'hand.seq{}.png'.format(i+1)))
    var.append(np.array(img)[:480:48, 10:490:48].flatten())
    
var = np.var(np.array(var))
dim = np.interp(var, centers, range(60))
print dim
"""
