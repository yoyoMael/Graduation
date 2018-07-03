#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

import re
import os.path as path



data_dir = "./dataset/vocabulary/"
if __name__ == "__main__":

  f = open('./dataset/vocabulary/GRE.txt', 'r')
  content = f.read()
  f.close()
  pattern = re.compile("Q: (\\b\\w+\\b)")
  vocabularies = re.findall(pattern,content)
  f = open(path.join(data_dir,"GRE_pure.txt"), 'w')
  vocabulary = '\n'.join(vocabularies)
  f.write(vocabulary)
  f.close()