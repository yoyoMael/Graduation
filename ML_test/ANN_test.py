# -*- coding:utf-8 -*-
__author__ = 'yoyo'


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

N = 60000
mnist = fetch_mldata("mnist-original",data_home = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/mnist_original')
x, y = mnist.data/255., mnist.target

print x[0]
x_train, x_test = x[0:N], x[N:]
y_train, y_test = y[0:N], y[N:]

mlp = MLPClassifier(hidden_layer_sizes = (50,), max_iter = 10, alpha = 1e-4,
 solver = 'sgd', verbose = 10, tol = 1e-4, random_state = 1, learning_rate_init = .1)

mlp.fit(x_train, y_train)
print "Training set score: %f " % mlp.score(x_train, y_train)
print "Test set score: %f" % mlp.score(x_test, y_test)

