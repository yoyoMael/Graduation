#! python2
# -*- conding:utf-8 -*-
from PIL import Image
import struct
from sklearn import tree
import pydotplus
from sklearn import cross_validation
import pickle
from sklearn.naive_bayes import GaussianNB



def read_image(filename):
  f = open(filename, 'rb')

  index = 0
  buf = f.read()

  f.close()

  magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
  # reading magic etc.
  index += struct.calcsize('>IIII')
  imgs = []

  for i in xrange(images):
    image = []
    # 'L' option represents a grey picture, second parameter is the size of image
    for x in xrange(rows):
      for y in xrange(columns):
        image.append(int(struct.unpack_from('>B', buf, index)[0]))
        # reading grey degree number and write into the image
        index += struct.calcsize('>B')
    imgs.append(image)
  return imgs

    # print 'save' + str(i) + 'image'
    # image.save(str(i) + '.png')
    # # save the image into a file

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


if __name__ == '__main__':
  x = read_image('../train-images-idx3-ubyte')
  y = read_label('../train-labels-idx1-ubyte')

  # dot_data = tree.export_graphviz(clf, out_file=None)
  # graph = pydotplus.graph_from_dot_data(dot_data)
  # graph.write_pdf("iris-dt.pdf")
  # print  cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=3)


