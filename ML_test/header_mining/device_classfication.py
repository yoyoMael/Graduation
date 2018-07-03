#! python2
# -*- coding:utf-8 -*-
__author__ = 'yoyo'


data_file = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/traindata.txt'

def data_genor(f):
  while(1):
    line = f.readline()
    if line:
      yield line
      continue
    else:
      break

if __name__ == '__main__':
  f = open(data_file)
  for x in data_genor(f):



