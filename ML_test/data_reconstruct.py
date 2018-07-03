#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'




if __name__ == '__main__':
  data_file = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/traindata.txt'
  f = open(data_file, 'r')
  i = 0
  out_f = open('/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/train_data.txt', 'w')
  for x in data_genor(f):
    reline = []
    elements = x.split('\t')
    labels = elements[1].split('|')
    features =
    reline.extend(labels)
    reline = '|'.join(reline)
    out_f.write(reline)
    i += 1
    print i
  out_f.close()
  f.close()








