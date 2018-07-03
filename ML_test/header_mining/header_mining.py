#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

from prefixspan import PrefixSpan as PS
from get_by_label import *
import os.path as path
import re



def get_longest(sequences):
  len_prev = 0
  index = 0
  for i in range(len(sequences)):
    len_current = len(sequences[i][1])
    if len_current > len_prev:
      index = i
      len_prev = len_current
    else:
      continue

  return sequences[index]
# get the longest common sequence in the given ps set

def base_sequence(dataset, num, support):
  data = []

  for i in range(0,num):
    item = dataset[i].split("\t")[0]
    data.append(item)
  ps_base = PS(data)
  base_sequence = get_longest(ps_base.topk(1000))

  return base_sequence
# dataset is the list of data
# num is the amount of data used to generate sequence
# support is the required support

def get_common(sequence, data):
  sequence = "".join(sequence[1])
  data = data.split("\t")[0]
  data = [sequence, data]
  ps = PS(data)
  common = get_longest(ps.topk(1000))
  print("Current sequence: ")
  print(common)
  return common

def sequence_mining(data, sequence, start, end):
  for i in range(start, 100):
    sequence = get_common(sequence, headers[i])
  return sequence

def match(data, sequence_string, i = 0):
  data = data.split("\t")[0]
  current_index = 0
  for i in len(sequence_string):
    if item in data:
      match(data[current_index:], sequence_string)


if __name__ == "__main__":
  data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/"
  filename = "iOS 6.1.txt"
  headers = get_lines(data_dir, filename)
  initial_num = 10
  end_mining = 1000
  base_sequence = base_sequence(headers, initial_num, 5)
  print(base_sequence)
  final_sequence = sequence_mining(headers, base_sequence, initial_num + 1, end_mining)
  final_sequence = "".join(final_sequence[1])







