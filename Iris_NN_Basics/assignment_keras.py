# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 22:00:06 2019

@author: Eric Lonewolf
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris.csv')
dataset = dataset.iloc[50:150,2:6]
dataset = pd.get_dummies(dataset, columns=['variety'])             # One Hot Encoding
values = list(dataset.columns.values)

Y = dataset[values[-2:]] 
Y = np.array(Y, dtype='float32')
X = dataset[values[0:-2]]
X = np.array(X, dtype='float32')

indices = np.random.choice(len(X), len(X), replace=False)          # Shuffling the Data
X_values = X[indices]
Y_values = Y[indices]

#%% Creating a Train and a Test Dataset 

test_size = 10                                                    
X_test = X_values[-test_size:]   
X_train = X_values[:-test_size]
Y_test = Y_values[-test_size:]
Y_train = Y_values[:-test_size]

model=tf.keras.Sequential()
# Add the layers
model.add(tf.keras.layers.Dense(6,input_dim=2,activation='sigmoid'))
model.add(tf.keras.layers.Dense(2,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
#%%
model.summary()
train=model.fit(X_train,Y_train,nb_epoch=10000,verbose=2)

print(model.predict(Y_train))

loss_curve=train.history['loss']
acc_curve=train.history['accuracy']
plt.plot(loss_curve,label='Loss')
plt.legend(loc='upper_left')

plt.plot(acc_curve, label='Accuracy')
plt.legend(loc='upper_rigupper_left')

score = model.evaluate(X_test, Y_test)
