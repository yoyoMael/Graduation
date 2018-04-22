# -*- coding:utf-8 -*-
#! python2
__author__ = 'yoyo'

import numpy as np

from PIL import Image

def load_feature(filename):
  data = np.load(filename)
  data = data / 255.
  return data

if __name__ == '__main__':
  for i in xrange(10):
    data = np.load(str(i) + 't10k.npy')
    mean = data[0]
    for j in range(1,len(data)):
      mean += data[j]
    mean = mean/len(data)
    mean = mean.reshape(28,28)
    image = Image.new('L', (28,28))
    for x in xrange(28):
      for y in xrange(28):
        pixel = int(mean[y][x])
        image.putpixel((x,y), pixel)
    image.save('mean' + str(i) + '.png')
    print 'created %d' % i




