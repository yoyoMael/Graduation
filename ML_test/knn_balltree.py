__authot__ = 'yoyo'
# -*- coding:utf-8 -*-

import sys
import urllib
import urlparse
import re
import numpy as np
from sklearn.externals import joblib
import HTMLParser
import nltk
import csv
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn import metrics

def load_cmd_seq_list(filename):
  cmd_seq_list = []
  dist = []
  with open(filename) as f:
    i = 0
    x = []
    for line in f:
      line = line.strip('\n')
      x.append(line)
      dist.append(line)
      i += 1
      if i == 100:
        cmd_seq_list.append(x)
        x = []
        i = 0
    fdist = FreqDist(dist).keys()
    dist_most = set(fdist[0:50])
    dist_least = set(fdist[-50:])
  return cmd_seq_list, dist_most, dist_least

def get_cmd_feature(cmd_seq_list, dist_most, dist_least):
  cmd_feature_list = []
  for cmd_seq in cmd_seq_list:
    f1 = len(set(cmd_seq))
    fdist = FreqDist(cmd_seq).keys()
    f2 = fdist[0:10]
    f3 = fdist[-10:]
    f2 = len(set(f2) & set(dist_most))
    f3 = len(set(f3) & set(dist_least))
    cmd_feature = [f1, f2, f3]
    cmd_feature_list.append(cmd_feature)
  return cmd_feature_list

def get_label(filename, user = 1):
  x = [0] * 50
  index = user - 1
  with open(filename) as f:
    for line in f:
      line = line.strip('\n')
      x.append(int(line.split()[index]))
  return x

N = 100

if __name__ == '__main__':
  score_list = []
  for i in range(1, 51):
    cmd_seq_list, dist_most, dist_least = load_cmd_seq_list('/Users/Yoyo/Desktop/Graduation/ML_test/dataset/masquerade-data/User'+str(i))
    cmd_feature_list = get_cmd_feature(cmd_seq_list, dist_most, dist_least)
    labels = get_label('/Users/Yoyo/Desktop/Graduation/ML_test/dataset/masquerade-data/label.txt', i)

    x_train = cmd_feature_list[0:N]
    y_train = labels[0:N]

    x_test = cmd_feature_list[N:]
    y_test = labels[N:]

    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    score = np.mean(y_predict == y_test)
    print score
    score_list.append(score)
  print 'Mean Score: ' + str(np.mean(score_list))






