#! python2
# -*- coding:utf-8 -*-

__author__ = 'yoyo'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(0)

x = np.r_[np.random.randn(100,2) - [2,2], np.random.randn(100,2) + [2,2]]
y = [0] * 20 + [1] * 20

clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5,5)
yy = a * xx - (clf.intercept_[0]) / w[1]
# x1 和 x2 用来画出超平面 w† * x + b = 0, x = [x1, x2]

sv = clf.support_vectors_[0]
yy_down = a * xx + (sv[1] - a*sv[0])
sv = clf.support_vectors_[-1]
yy_up = a * xx + (sv[1] - a * sv[0])

plt.plot(xx, yy, 'c-')
plt.plot(xx, yy_down, 'm-')
plt.plot(xx, yy_up, 'r-')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 200, facecolors = 'none')
# s is the size of the circle, facecolor is the color of the circle
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Paired)
plt.axis('tight')
plt.show()

