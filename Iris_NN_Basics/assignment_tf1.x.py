import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np

#%% Subsetting the dataset

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

test_size = 50                                                    
X_test = X_values[-test_size:]   
X_train = X_values[:-test_size]
Y_test = Y_values[-test_size:]
Y_train = Y_values[:-test_size]

#%% Initializing parameters

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
interval = 1000
epoch = 10000
n_h = 6
n_x = 2
n_y = 2

X_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
Y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)

w1 = tf.Variable(tf.random_normal(shape=[n_x,n_h])) 
b1 = tf.Variable(tf.random_normal(shape=[n_h]))   
w2 = tf.Variable(tf.random_normal(shape=[n_h,n_y])) 
b2 = tf.Variable(tf.random_normal(shape=[n_y])) 

#%% Forward Propagation

Z1 = tf.nn.tanh(tf.add(tf.matmul(X_data, w1), b1))
Z2 = tf.nn.sigmoid(tf.add(tf.matmul(Z1, w2), b2))
final_output = tf.nn.softmax(Z2)

#%% Cost Function

cost = tf.reduce_mean(-tf.reduce_sum(Y_target * tf.log(final_output), axis=0))

#%% Backward Propagation

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)   #Optimiser

#%% Initialize variables

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#%% Training Model

print('Model Training')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, Y_target: Y_train})
    if i % interval == 0:
        print('Epoch', i, '|', 'Cost:', sess.run(cost, feed_dict={X_data: X_train, Y_target: Y_train}))

#%% Prediction
        
print()
for i in range(len(X_test)):
    print('Actual:', Y_test[i], 'Predicted:', np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))
