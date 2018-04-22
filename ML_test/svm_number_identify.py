# -*- coding:utf-8 -*-
#! python2
__author__ = 'yoyo'

import struct
from sklearn import svm
import pickle
import numpy as np

dirct = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/MNIST'

def read_image(filename):
  # read images in the form of a normalized number sequence, save in a list
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
    print 'read image..'
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
    print 'read label..'

  return labelArr

N = 7000
if __name__ == '__main__':
  images = read_image(dirct+'/t10k-images-idx3-ubyte')
  labels = read_label(dirct+'/t10k-labels-idx1-ubyte')
  x_train, x_test = images[0:N], images[N:]
  y_train, y_test = labels[0:N], labels[N:]
  clf = svm.SVC(decision_function_shape='ovr', kernel = 'linear')
  # ovr use one class as positive, others as negative data, and ovo use pairs of different classes to train
  # ovr has only n clfs, but cost more time computing, ovo has n(n-1)/2 clfs, less computing time
  clf.fit(x_train, y_train)
  svm_clf = pickle.dumps(clf)
  f = open('svm_number_clf.model','w')
  f.write(svm_clf)
  f.close()
  print clf.score(x_test, y_test)



