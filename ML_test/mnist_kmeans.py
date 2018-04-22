# -*- coding:utf-8 -*-
#! python2

import numpy as np
import math

import matplotlib.pyplot as plt

N_measure = 3000
N_train = 5000
N_test = 2000

def get_measure(data):
  mean = data[0]
  for j in range(1,len(data)):
    mean += data[j]
  mean = mean/len(data)
  return mean

def load_numbers(filename):
  data = np.load(filename)
  return data

def distance(x1, x2):
  diff = x1 - x2
  distance = np.mean(np.multiply(diff, diff))
  return distance

def coordnation(x,center):
  x = [distance(x,)]

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

if __name__ == '__main__':
  data = []
  center_data = []
  center = []

  for i in xrange(10):
    data.append(np.load(str(i) + 't10k.npy')/255.)
    center_data.append(data[i][0:299])
  for i in xrange(10):
    center.append(get_measure(center_data[i]))
  plt.figure(figsize = [16,8])
  for n in xrange(10):
    distances = []
    data0 = data[n][300:]
    distance1 = 0
    for i in xrange(10):
      for j in xrange(len(data0)):
        distance1 += distance(data0[j], center[i])
      distance1 = distance1/len(data0)
      distances.append(distance1)
    plt.subplot(2,5,n+1)
    plt.bar(range(0,10), distances, align = 'center', color = 'g')
    print n,distances
  plt.show()




