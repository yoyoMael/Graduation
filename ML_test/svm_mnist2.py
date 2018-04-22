# -*-coding:utf-8 -*-
#! python2
__author__ = 'yoyo'

import numpy as np
import scipy
import math

from matplotlib import pyplot as plt
from sklearn import svm

def load_feature(filename):
  data = np.load(filename)
  data = data / 255.
  return data

class svm_mnist(svm.SVC):
  def __init__(self, data_train, label_train, data_test, lable_test, C = 1.1, kernel = 'rbf', decision_function_shape = 'ovr'):
    svm.SVC.__init__(self, C = 1.1, kernel = 'rbf', decision_function_shape = 'ovr')
    self.data_train = data_train
    self.data_test = data_test
    self.label_test = label_test
    self.label_train = label_train

  def classify(self):
    self.fit(self.data_train, self.label_train)
    # self.clf.test(self.data_test, self.label_test)
    print self.score(self.data_test, self.label_test)

if __name__ == '__main__':
  data0 = load_feature('0t10k.npy')
  data1 = load_feature('1t10k.npy')
  data2 = load_feature('2t10k.npy')
  data_train = np.r_[data0[0:600], data1[0:300], data2[0:300]]
  label_train = [0] * 600 + [1] * 300 + [1] * 300
  data_test = np.r_[data0[600:], data1[600:]]
  label_test = [0] * (len(data0) - 600) + [1] * (len(data1) - 600)

  a = svm_mnist(data_train, label_train, data_test, label_test)
  a.classify()








