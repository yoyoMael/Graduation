#! python2
#! -*- conding:utf-8 -*-
import sys
import urllib
import urlparse
import re
import numpy as np
import nltk
from nltk import FreqDist
import csv
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from math import log

N = 199
neg_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moviedata/review_polarity/txt_sentoken/neg'
pos_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moviedata/review_polarity/txt_sentoken/pos'
P_pos = -1
P_neg = -1
least_limit = 2
least_len = 2
common_words = ['this', 'that', 'the', 'are', 'is']

def get_train_text(dir):
  text_list = os.listdir(dir)[0:N]
  for file in text_list:
    text_list[text_list.index(file)] = os.path.join(dir, file)
  text = ''
  for x in text_list:
    f = open(x,'rb')
    buf = f.read()
    f.close()
    text += buf
  text = text.strip(' ')
  return text
  # adding all training files into one

def get_words(text):
  lines = text.split('\n')
  buf = []
  for line in lines:
    line = line.strip(' ')
    sentences = line.split('.')
    for sentence in sentences:
      sentence = sentence.strip(' ')
      words = sentence.split(' ')
      for word in words:
        if len(word) > least_len and word not in common_words:
          buf.append(word)
  return buf
# buf is the list of all words, len(buf) is the total number of words

# def get_train_text_len(file)
#   f = open(file, 'rb')
#   buf = f.read()
#   f.close()
#   buf = buf.strip(' ')
#   words = buf.split(' ')
#   return len(words)

def get_p_words(vocabulary, neg_dist, pos_dist, all_neg_words, all_pos_words):
  p_neg = dict()
  p_pos = dict()
  for word in vocabulary:
    if word in all_neg_words:
      p_neg[word] = log((neg_dist[word] + 1)/float(len(vocabulary)+len(all_neg_words)))
  for word in vocabulary:
    if word in all_pos_words:
      p_pos[word] = log((pos_dist[word] + 1)/float(len(vocabulary)+len(all_pos_words)))
  return p_neg, p_pos
# computing the probability of all words

def cut_dist(dist):
  no_use = []
  for x in dist:
    if dist[x] <= least_limit:
      no_use.append(x)
  for x in no_use:
    dist.pop(x)
  return dist
# this function is used to delete rare words


if __name__ == '__main__':
  neg_train_text = get_train_text(neg_dir)
  pos_train_text = get_train_text(pos_dir)
  all_neg_words = get_words(neg_train_text)
  all_pos_words = get_words(pos_train_text)
  vocabulary = set(all_neg_words + all_pos_words)
  neg_dist = FreqDist(all_neg_words)
  pos_dist = FreqDist(all_pos_words)
  neg_dist = cut_dist(neg_dist)
  pos_dist = cut_dist(pos_dist)
  pos_num = 0
  neg_num = 0
  p_neg_word, p_pos_word = get_p_words(vocabulary, neg_dist, pos_dist, all_neg_words,all_pos_words)
  for i in range(N+1,300):
    f = open(os.path.join(pos_dir, os.listdir(pos_dir)[i]), 'rb')
    buf = f.read()
    f.close()
    buf.strip(' ')
    test_words = get_words(buf)
    p_test_neg = P_pos
    p_test_pos = P_neg

    type = 'oh!'
    for word in test_words:
      if word in neg_dist:
        p_test_neg += p_neg_word[word]
    for word in test_words:
      if word in pos_dist:
        p_test_pos += p_pos_word[word]

    if p_test_pos > p_test_neg:
      type = 'pos'
      pos_num += 1
    else:
      type = 'neg'
      neg_num += 1
    print type, p_test_pos, p_test_neg
  print "pos: " , pos_num , "neg: " , neg_num










