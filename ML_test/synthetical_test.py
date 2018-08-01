#! python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.model_selection import ShuffleSplit
from matplotlib.colors import ListedColormap
from lightgbm.sklearn import LGBMClassifier

import warnings

def plot_learning_curve(clf, title, data, labels, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1,1.0,5)):
  plt.figure()
  plt.title(title)
  if ylim is not None:
    plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes, train_scores, test_scores = learning_curve(clf, data, labels, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  print(train_scores)
  print(test_scores)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  plt.grid()

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color = "r")

  plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, '-', color="r", label="Training score")
  plt.plot(train_sizes, test_scores_mean, '-', color="g", label="Cross_validation score")

  plt.legend(loc="best")

  return plt


warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# x, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
#                            n_redundant=0, n_repeated=0,
#                            random_state=42, n_clusters_per_class=1)
x, y = make_moons(noise=0.3, random_state=0)
lgb = LGBMClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=2)

lgb.fit(x_train, y_train)

# plot
figure = plt.figure(figsize=(10,8))
x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))
print(xx)
cm = plt.cm.RdBu
cm_bright = plt.cm.RdBu
ax = plt.subplot(1,1,1)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright,
           alpha = .6, edgecolors='k')




z = lgb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

z=z.reshape(xx.shape)

ax.contourf(xx, yy, z, cmap=cm, alpha=.8)



plt.show()