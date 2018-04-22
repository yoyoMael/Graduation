# -*- coding:utf-8 -*-
#! python2

import struct
import numpy as np
import pickle

from PIL import Image

dirct = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/MNIST/'

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

def arrange():
  labels = read_label(dirct + 't10k-labels-idx1-ubyte')
  f = open(dirct + 't10k-images-idx3-ubyte', 'rb')
  index = 0
  buf = f.read()
  f.close()

  N_test = 10

  buff = []
  # buff stores the different in 10 lists
  for i in xrange(10):
    buff.append([])

  magic, img_num, rows, columns = struct.unpack_from('>IIII', buf, index)
  # 读取魔数等
  index += struct.calcsize('>IIII')
  # 增加索引值
  for i in xrange(img_num):
    image = []
    n = 0
    for x in xrange(rows):
      for y in xrange(columns):
        pixel = int(struct.unpack_from('>B', buf, index)[0])
        image = np.append(image, pixel)
        # 读取灰度值，存入矩阵中
        index += struct.calcsize('>B')
    buff[labels[i]].append(image)
    # store datas according to the label by using labels[i]
    print 'Appended' + str(labels[i])


  buf = []
  # no need to use buf, so I reset it to []
  for img in buff:
    buf.append(np.array(img))
    # transform mnist datas into array, so we can use np.save()
  return buf


def write_files(buf):
  for i in xrange(len(buf)):
    np.save(str(i), buf[i])

if __name__ == '__main__':
  buf = arrange()
  write_files(buf)
