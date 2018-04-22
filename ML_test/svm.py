# -*-coding:utf-8 -*-
#! python2
__author__ = 'yoyo'

import numpy
import scipy
import math

from matplotlib import pyplot as plt


N = 200
d = 1.0
# number of points of each cluster
# the variance of each cluster, here all the variances are identical
np.random.seed(0)
x = np.r_[d * np.random.randn(N,2) - [1.5,1.5], d * np.random.randn(N,2) + [1.5,1.5]]
y = [0] * N + [1] * N + [2] * N
# generate the different distribution of data, and corresponding labels
y = np.array(y)
z = y
# z is the latent variable



