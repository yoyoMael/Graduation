#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

import os.path as path

data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database"
filename = "traindata.txt"

if __name__ == "__main__":
  filepath = path.join(data_dir, filename)
  f = open(filepath)
  headers = f.readlines()
  f.close()
  generic_ios = []
  for header in headers:
    if "Generic iOS" in header:
      generic_ios.append(header)
  f = open(path.join(data_dir,"generic_ios.txt"),"w")
  f.write("".join(generic_ios))
  f.close()


