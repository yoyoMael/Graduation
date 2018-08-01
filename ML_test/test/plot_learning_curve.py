#! python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from lightgbm import LGBMClassifier
from sklearn.datasets import load_digits

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


if __name__ == "__main__":
  digits = load_digits()
  train_data, train_labels = digits.data, digits.target
  clf = gnb()
  cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
  title = "Learning Curves(Gaussian NB)"
  plot_learning_curve(clf, title, train_data, train_labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
  clf = SVC(gamma=0.001)
  title = "Learning Curves(SVM)"
  plot_learning_curve(clf, title, train_data, train_labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
  print("Figure finished")
  plt.show()