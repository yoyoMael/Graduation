#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

from prefixspan import PrefixSpan as PS
import os.path as path

data_dir = "./dataset/vocabulary/"
filename = "GRE_pure.txt"

if __name__ == "__main__":
  filepath = path.join(data_dir, filename)
  f = open(filepath)
  vocabulary = f.read()
  vocabulary = vocabulary.split("\n")
  f.close()
  ps = PS(vocabulary)
  for sequence in ps.frequent(3):
    if len(sequence[1]) >= 4:
      print(sequence)
