#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'

import math
import numpy as np

N = 200
d = 1.0
# number of points of each cluster
# the variance of each cluster, here all the variances are identical
np.random.seed(0)
x = np.r_[d * np.random.randn(N,2) - [1.5,1.5], d * np.random.randn(N,2) + [1.5,1.5]]
y = [0] * N + [1] * N + [2] * N + [2] * 4
# generate the different distribution of data, and corresponding labels
y = np.array(y)
z = y
# z is the latent variable

def distance2(a, b):
  return math.pow((a[0]-b[0]),2) + math.pow((a[1] - b[1]),2)

def k_neighbors(new, dataset, k):
  distance_list = []
  for point in dataset:
    distance_list.append(distance2(new, point))
  list_copy = distance_list[:]
  distance_list.sort()
  k_neighbors = []
  for nmax in distance_list[:k]:
    k_neighbors.append(list_copy.index(nmax))

  return k_neighbors

if __name__ == '__main__':
  neighbors = k_neighbors([0,.1], x, 10)
  for n in neighbors:
    print x[n], y[n]





