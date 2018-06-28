#! python2
# -*- coding:utf-8 -*-

import math
from math import log
import numpy as np
from matplotlib import pyplot as plt

N = 200
d = 1.0
# number of points of each cluster
# the variance of each cluster, here all the variances are identical
np.random.seed(0)
x = np.random.binomial(1, .5, (1,4))
y = [0] * N
for
# generate the different distribution of data, and corresponding labels
y = np.array(y)
data = np.c_[x, y]
# z is the latent variable



def entropy(dataset):
  num = len(dataset)
  labelcounts = {}
  for vect in dataset:
    label = vect[-1]
    if label not in labelcounts.keys():
      labelcounts[label] = 0
      labelcounts[label] += 1
  entropy = .0
  for key in labelcounts:
    prob = labelcounts[key]/float(num)
    entropy -= prob * log(prob, 2)
  return entropy

def createDataSet():
  dataSet = [[1, 1, 'yes'],
  [1, 1, 'yes'],
  [1, 0, 'no'],
  [0, 1, 'no'],
  [0, 1, 'no']]

def splitDataSet()




