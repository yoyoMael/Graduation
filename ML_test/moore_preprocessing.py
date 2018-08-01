#! python3
# -*- coding:utf-8 -*-

import os

data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore/'
prefix = 'entry'

def to_str(num):
  if num < 10:
    string = '0' + str(num)
  else:
    string = str(num)
  return string

if __name__ == '__main__':
  for suffix in range(10):
    filename = prefix + to_str(suffix)
    filedir = os.path.join(data_dir, filename)
    if os.path.exists(filedir):
      f = open(filedir, 'r')
      data = f.readlines()
      f.close()
      data = data[253:]
      f = open(filedir, 'w')
      for line in data:
        f.write(line)
      f.close()
      print("Finished %s"%filename)



