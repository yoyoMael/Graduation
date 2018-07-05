#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

import os.path as path



def get_lines(data_dir, filename):
  filepath = path.join(data_dir, filename)
  f = open(filepath)
  contents = f.readlines()
  f.close()
  return contents
  # open file an get the content

def extract_by_label(label, label_num, contents):
  data = []
  for content in contents:
    line = content.split("\t")
    if len(line) > 1:
      labels = line[1].split("|")
      if len(labels) > label_num:
        if label in labels[label_num - 1].strip("\n"):
          data.append(content)
  return data
# specify the label and the number of label(3 types of label in the data)

def write_file(filename, data, symbol=""):
  if type(symbol) == str:
    f = open(filename, "w")
    f.write(symbol.join(data))
    f.close()
  else:
    print("Not a usable joint symbol")
# write file, by specify the joint symbol

if __name__ == "__main__":

  data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database"
  filename = "traindata.txt"
  label = "iOS 8.0"
  label_num = 2
  headers = get_lines(data_dir, filename)
  data = extract_by_label(label, 2, headers)
  write_file(path.join(data_dir, label + ".txt"), data)



