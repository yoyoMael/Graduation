#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'

N = 200
d = 1.0
# number of points of each cluster
# the variance of each cluster, here all the variances are identical
np.random.seed(0)
x = np.r_[d * np.random.randn(N,2) - [1.5,1.5], d * np.random.randn(N,2) + [1.5,1.5]]
x = np.r_[x, d * np.random.randn(N,2) + [1.5, -1.5]]
x = np.r_[x, [10, 20], [50]]
print x
y = [0] * N + [1] * N + [2] * N + [2] * 4
# generate the different distribution of data, and corresponding labels
y = np.array(y)
z = y
# z is the latent variable

def distance2(a, b):
  return math.pow((a[0]-b[0]),2) + math.pow((a[1] - b[1]),2)

