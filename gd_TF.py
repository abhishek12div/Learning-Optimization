#TRAINING sine-function using GRADIENT DESCENT

import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

'''def normalizei(data):
    m1 = min(data)
    m2 = max(data)
    diff = m2 - m1
    y = data - m1
    a = -1
    b = 1
    X = a +((y*(b-a))/diff)
    return X'''

def normalize(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    #denom[denom==0] = 1
    return x_min + nom/denom#, X.min(axis=0), X.max(axis=0)

def denormalize(X, x_min, x_max):
    return X*(x_max-x_min)+x_min

#TRAIN DATA        ####################################################
X_train = np.arange(-4,4,0.02)
Y_train = 1+(X_train + 2 * X_train**2)*np.sin(-X_train**2)
print('train data shape', X_train.shape)
X = normalize(X_train, -1, 1)
print('normalized train data shape', X.shape)
X = X.reshape(len(X), 1)
#print(X.shape)

Y = normalize(Y_train, -1, 1)
Y = Y.reshape(len(Y), 1)

#######################################################################

#TEST DATA       ######################################################
X_test = np.random.uniform(-4,4,10000)
Y_test = 1 + (X_test+2*X_test*X_test)*np.sin(-X_test*X_test)

X1= normalize(X_test, 0, 1)
X1 = X1.reshape(len(X1), 1)

Y1 = normalize(Y_test, 0, 1)
Y1 = Y1.reshape(len(Y1), 1)

#######################################################################

input_nodes = X.shape[1]
h1 = 7
#h2 = 18
#h3 = 8
output_nodes = Y.shape[1]

seed = np.random.randint(0,100)
tf.set_random_seed(seed)
print(seed)

data = tf.placeholder(dtype=tf.float32, shape=[None, input_nodes])
label = tf.placeholder(dtype=tf.float32, shape=[None, output_nodes])

fully_connected1 = tf.contrib.layers.fully_connected(inputs=data,
                                                     num_outputs=h1,
                                                     activation_fn=tf.nn.sigmoid,
                                                     weights_initializer=tf.random_uniform_initializer(-0.5,0.5),
                                                     biases_initializer=tf.random_uniform_initializer(-0.5,0.5)
                                                     )

#1st hidden to 2nd
'''fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1,
                                                     num_outputs=h2,
                                                     activation_fn=tf.nn.sigmoid,
                                                     weights_initializer=tf.random_uniform_initializer(-0.5, 0.5),
                                                     biases_initializer=tf.random_uniform_initializer(-0.5, 0.5)
                                                     )'''

#2nd to 3rd
'''fully_connected3 = tf.contrib.layers.fully_connected(inputs=fully_connected2,
                                                     num_outputs=h3,
                                                     activation_fn=tf.nn.sigmoid,
                                                     weights_initializer=tf.random_uniform_initializer(-0.5, 0.5),
                                                     biases_initializer=tf.random_uniform_initializer(-0.5, 0.5)
                                                     )'''

#to ouput
pred = tf.contrib.layers.fully_connected(inputs=fully_connected1,
                                         num_outputs=output_nodes,
                                         activation_fn=None,
                                         weights_initializer=tf.random_uniform_initializer(-0.5, 0.5),
                                         biases_initializer=tf.random_uniform_initializer(-0.5, 0.5)
                                         )

error_func = 0.5 * tf.reduce_mean(tf.square(pred - label))

model_sgd = tf.train.GradientDescentOptimizer(0.01).minimize(error_func)
max_iters = 10000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    cf_sgd = []
    for i in range(max_iters):
        result, err_sgd = sess.run([model_sgd, error_func], feed_dict={data: X, label: Y})
        if i % 1000 == 0:
            print("Cost after iteration %i %f" % (i, err_sgd))
        cf_sgd.append(err_sgd)
        
       
plt.semilogy(cf_sgd, label='GD')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


