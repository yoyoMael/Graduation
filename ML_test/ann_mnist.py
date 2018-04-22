#! python2
# -*- coding:utf-8 -*-

import struct
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier


dirct = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/MNIST'

def read_image(filename):
  f = open(filename, 'rb')

  index = 0
  buf = f.read()
  f.close()

  images = []
  magic, img_num, rows, columns = struct.unpack_from('>IIII', buf, index)
  # 读取魔数等
  index += struct.calcsize('>IIII')
  # 增加索引值
  for i in xrange(img_num):
    image = np.zeros(rows*columns)
    n = 0
    for x in xrange(rows):
      for y in xrange(columns):
        pixel = int(struct.unpack_from('>B', buf, index)[0])/255.
        image = np.append(image, pixel)
        # 读取灰度值，存入矩阵中
        index += struct.calcsize('>B')
    images.append(image)
  return images

def read_label(filename):
  f = open(filename, 'rb')
  index = 0
  buf = f.read()

  f.close()

  magic, labels = struct.unpack_from('>II', buf, index)
  index += struct.calcsize('>II')

  labelArr = [0] * labels

  for x in xrange(labels):
    labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
    index += struct.calcsize('>B')

  return labelArr

N_train = 7000
dirct = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/MNIST'

if __name__ == '__main__':
  images = read_image(dirct+'/t10k-images-idx3-ubyte')
  labels = read_label(dirct+'/t10k-labels-idx1-ubyte')
  x_train, x_test = images[0:N_train], images[N_train:]
  y_train, y_test = labels[0:N_train], labels[N_train:]

  mlp = MLPClassifier(hidden_layer_sizes = (50,), max_iter = 10, alpha = 1e-4,
 solver = 'sgd', verbose = 10, tol = 1e-4, random_state = 1, learning_rate_init = .1)
  mlp.fit(x_train, y_train)

  print "Training set score: %f " % mlp.score(x_train, y_train)
  print "Test set score: %f" % mlp.score(x_test, y_test)

