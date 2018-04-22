# -*-coding:utf-8 -*-
#! python2
__author__ = 'yoyo'

import numpy as np
import scipy
import math

from matplotlib import pyplot as plt
from sklearn import svm


# N = 200
# d = 1.0
# # number of points of each cluster
# # the variance of each cluster, here all the variances are identical
# np.random.seed(0)
# x = np.r_[d * np.random.randn(N,2) - [1.5,1.5], d * np.random.randn(N,2) + [1.5,1.5]]
# y = [0] * N + [1] * N + [2] * N
# # generate the different distribution of data, and corresponding labels
# y = np.array(y)
# z = y
# # z is the latent variable

def load_feature(filename):
  data = np.load(filename)
  data = data / 255.
  return data

# class svm_clf(svm.SVC):
#   def __init__(self, data_train, labels_train):
#     svm.SVC.__init__(self)
#     self.data = data_train
#     self.label = labels_train

#   def fit_data():
#     return self.fit(self.data, self.label)

#   def scoring():









if __name__ == '__main__':
  data0 = load_feature('0t10k.npy')
  data1 = load_feature('1t10k.npy')
  data_train = np.r_[data0[0:600], data1[0:600]]
  labels_train = [0] * 600 + [1] * 600
  data_test = np.r_[data0[600:], data1[600:]]
  labels_test = [0] * (len(data0) - 600) + [1] * (len(data1) - 600)
  clf = svm.SVC(C = 1.1, kernel = 'rbf',  decision_function_shape = 'ovr')
  clf.fit(data_train, labels_train)
  score = clf.score(data_test, labels_test)
  print 'Standard score(ovr, C=1.1, kernel = rbf): %f' % (score)
  # comparing kernels
  clf = svm.SVC(C = 1.1, kernel = 'linear', decision_function_shape = 'ovr')
  clf.fit(data_train, labels_train)
  score = clf.score(data_test, labels_test)
  print 'Linear kernel: %f' % (score)
  clf = svm.SVC(C = 1.1, kernel = 'poly', decision_function_shape = 'ovr')
  clf.fit(data_train, labels_train)
  score = clf.score(data_test, labels_test)
  print 'Polynomial kernel: %f' % (score)
  clf = svm.SVC(C = 1.1, kernel = 'sigmoid', decision_function_shape = 'ovr')
  clf.fit(data_train, labels_train)
  score = clf.score(data_test, labels_test)
  print 'Sigmoid kernel: %f' % (score)

  # comparing C
  C = [.01, .1, 2, 10, 20, 50, 100, 1000]
  for i in xrange(8):
    clf = svm.SVC(C = C[i], kernel = 'rbf', decision_function_shape = 'ovr')
    clf.fit(data_train, labels_train)
    score = clf.score(data_test, labels_test)
    print 'Score(C = %f): %f' % (C[i], score)







