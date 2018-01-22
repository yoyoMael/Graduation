import struct
import sklearn
from sklearn import tree
import pickle
from number_identify_tree_training import *

# def load_tree(name):
#   f = open(name, 'rb')
#   the_tree = f.read()
#   f.close()
#   the_tree = pickle.loads(the_tree)
#   return the_tree

# def read_image(filename):
#   f = open(filename, 'rb')

#   index = 0
#   buf = f.read()

#   f.close()

#   magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
#   # reading magic etc.
#   index += struct.calcsize('>IIII')
#   imgs = []

#   for i in xrange(images):
#     image = []
#     # 'L' option represents a grey picture, second parameter is the size of image
#     for x in xrange(rows):
#       for y in xrange(columns):
#         image.append(int(struct.unpack_from('>B', buf, index)[0]))
#         # reading grey degree number and write into the image
#         index += struct.calcsize('>B')
#     imgs.append(image)
#   return imgs

#     # print 'save' + str(i) + 'image'
#     # image.save(str(i) + '.png')
#     # # save the image into a file

# def read_label(filename):
#   f = open(filename, 'rb')
#   index = 0
#   buf = f.read()

#   f.close()

#   magic, labels = struct.unpack_from('>II', buf, index)
#   index += struct.calcsize('>II')

#   labelArr = [0] * labels

#   for x in xrange(labels):
#     labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
#     index += struct.calcsize('>B')

#   return labelArr

if __name__ == '__main__':
  x = read_image('../t10k-images-idx3-ubyte')
  y = read_label('../t10k-labels-idx1-ubyte')
  # z = read_image('../train-images-idx3-ubyte')
  clf1 = load_tree('the_tree')
  test = clf.predict(x)
  clf2 =
#compare the result of predict with the label file
  index = 0
  difference = 0
  for i in y:
    if test[index] != i:
      difference += 1
    index += 1

  print difference

