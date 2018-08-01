#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'

import numpy as np
from matplotlib import pyplot as plt
import math
from multiprocessing import Process, Queue, Pool, Manager

def distance2(a, b):
  return math.pow((a[0]-b[0]),2) + math.pow((a[1] - b[1]),2)
# define the distance measure in this model

def get_labels(centers):
  i = 0
  labels = []
  for center in centers:
    labels.append(i)
    i += 1
  return labels
# get the number of labels, labels are natural numbers

def clf(a, centers, labels):
  labels = []
  distances = []
  for center in centers:
    distances.append(distance2(a, center))
  min_distance = min(distances)
  for i in range(0,len(distances)):
    if distances[i] == min_distance:
      return i
# get the cluster that the point belongs to



N = 200
d = 1.0
# number of points of each cluster
# the variance of each cluster, here all the variances are identical
np.random.seed(0)
x = np.r_[d * np.random.randn(N,2) - [1.5,1.5], d * np.random.randn(N,2) + [1.5,1.5]]
x = np.r_[x, d * np.random.randn(N,2) + [1.5, -1.5]]
y = [0] * N + [1] * N + [2] * N
# generate the different distribution of data, and corresponding labels
y = np.array(y)
z = y
# z is the latent variable

means = np.random.randn(3,2)
# this is the initial centers of the clusters


def new_mean(x, centers, labels):
  z = []
  n = [0] * len(labels)
  sums = np.zeros((len(labels),2))
  for sum in sums:
    sum = np.random.randn(1,2)[0]
  for point in x:
    z.append(clf(point, centers, labels))
  for i in range(0, len(z)):
    sums[z[i]] += x[i]
    n[z[i]] += 1
  means = sums
  for i in range(0, len(means)):
    means[i] = sums[i]/n[i]

  return means, np.array(z)



plt.figure(2)
labels = get_labels(means)


for i in range(0,10):
  means, z= new_mean(x, means, labels)
  plt.scatter(x[:, 0], x[:, 1], c = z, cmap = plt.cm.Paired)
  plt.scatter([1.5, -1.5, 1.5],[1.5, -1.5, -1.5], s = 50, facecolors = 'r')
  plt.scatter(means[:,0], means[:,1], s = 50, facecolors = 'b')
  plt.axis('tight')
  plt.show()





