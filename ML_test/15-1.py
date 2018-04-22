#! python2
# -*- coding:utf-8 -*-

import tensorflow as tf
import pickle
import gzip

def label_coding(label, size = 10):
  vector = []
  for num in label:
    v = [0] * size
    v[(num-1)] = 1
    vector.append(v)
  return vector
# turn numbers into one-hot code (scalar to 10d vector)

def load_data(filename):
  with gzip.open(filename) as f:
    data_train, data_valid, data_test = pickle.load(f)
    return data_train, data_valid, data_test

if __name__ == 'main':
  data_train, data_valid, data_test = load_data('./dataset/MNIST/mnist.pkl.gz')

  data_train, label_train = data_train
  data_test, label_test = data_test
  label_train = label_coding(label_train)
  label_test = label_coding(label_test)
  batch_size = 100

  x = tf.placeholder('float', [None, 784])

  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  y = tf.nn.softmax(tf.matmul(x, w) + b)
  y_ = tf.placeholder("float", [None, 10])

  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(int(len(data_train)/batch_size)):
    batch_xs = data_train[(i*batch_size):((i+1)*batch_size)]
    batch_ys = label_train[(i*batch_size):((i+1)*batch_size)]

    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print(sess.run(accuracy, feed_dict={x: x1, y_: y1}))

