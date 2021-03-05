# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:59:17 2019

@author: Eric Lonewolf
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#%% Asignment 1
#Q1

data = pd.read_csv('Iris.csv') 
print(data.shape)
for i in range(5):
    print(data.columns[i],"  ",data.dtypes[i])
    print("\n")

#Q2

X= data ['sepal.length'].values
Y= data ['sepal.width'].values

plt.title('Sepal',fontsize=20)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.scatter(X,Y)

#Q3

data.plot.box()


#%% Asignment 2
#Q1

#%%    Subsetting Dataset

data1 = data.copy()                                                            
data1.loc[data1['variety']=='Setosa','species'] = 2
data1.loc[data1['variety']=='Versicolor','species']=1
data1.loc[data1['variety']=='Virginica','species']=0
data1 = data1[data1['species']!= 2]
data1 = data1.drop(['sepal.length','sepal.width'],axis=1)      
                
#%%    PLotting a Scatter plot of required features

X = data1[['petal.length','petal.width']].values.T
Y = data1[['species']].values.T
plt.title('Iris Plot Versicolor - Black , Red - Virginica ',fontsize='15')
plt.xlabel('Petal Length',fontsize = '12')
plt.ylabel('Petal Width',fontsize = '12')
plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=50, cmap='RdGy');

array = data1.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
#%%    Initializing parameters

def init_p(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))  
    W2 = np.random.randn(n_y, n_h) * 0.01  
    b2 = np.zeros(shape=(n_y, 1)) 
       
    p = {"W1": W1,
         "b1": b1,
         "W2": W2,
         "b2": b2}
    
    return p

#%%    Setting the size of input, hidden, output layers
    
def ls(X, Y):
    n_x = X.shape[0]                                             
    n_h = 6                                                       
    n_y = Y.shape[0]                                              
    return (n_x, n_h, n_y)

#%%    Forward Propagation
    
def f_p(X, p):
    W1 = p['W1']
    b1 = p['b1']
    W2 = p['W2']
    b2 = p['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)                   #tanh activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = 1/(1+np.exp(-Z2))             #sigmoid activation function
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

#%%    Calculating Cost
    
def Cost(A2, Y, p):
   
    m = Y.shape[0] 
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))    
    cost = - np.sum(logprobs) / m               #cross-entropy cost
    
    return cost

#%%    Backward Propagation
    
def b_p(p, cache, X, Y):

    m = X.shape[1]
    
    W1 = p['W1']
    W2 = p['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
                                                                 
    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

#%%     Updating Parameters
    
def u_p(p, grads, learning_rate=1.2):

    W1 = p['W1']
    b1 = p['b1']
    W2 = p['W2']
    b2 = p['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
     
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    p = {"W1": W1,
         "b1": b1,
         "W2": W2,
         "b2": b2}
    
    return p

#%%      Creating Neural Network Model
    
def nn_model(X, Y, n_h, epoch=10000, print_cost=False):
    np.random.seed(3)
    n_x = ls(X, Y)[0]
    n_y = ls(X, Y)[2]
    
    p = init_p(n_x, n_h, n_y)
                                               
    for i in range(0, epoch):                 
             
        A2, cache = f_p(X, p)                  # Forward propagation
        cost = Cost(A2, Y, p)                  # Cost function 
        grads = b_p(p, cache, X, Y)            # Back propagation 
        p = u_p(p, grads)                      # Gradient descent parameter update
            
        if print_cost and i % 1000 == 0:       # Print the cost every 1000 epoch
            print ("Epoch %i: Cost %f" % (i, cost))
    return p,n_h


##### paste this in console     p = nn_model(X,Y , n_h = 6, epoch=10000, print_cost=True)