#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

from prefixspan import PrefixSpan as PS
from get_by_label import *
import os.path as path
import random



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

  return "".join(sequences[index][1])
# get the longest common sequence in the given ps set

def base_sequence(dataset, num):
  data = []

  for i in range(0,num):
    item = dataset[i].split("\t")[0].lower()
    data.append(item)
  ps_base = PS(data)
  base_sequence = get_longest(ps_base.topk(1000))
  return base_sequence
# dataset is the list of data
# num is the amount of data used to generate sequence

def get_common(sequence, data):
  data = data.split("\t")[0].lower()
  data = [sequence, data]
  ps = PS(data)
  common = get_longest(ps.topk(1000))
  return common

def get_seq_list(data, initial_num, group_size, total):
  used = total
  copy = []
  for x in data:
    copy.append(x)
  seq_list = []
  for i in range(0,total,group_size):
    group = []
    random.shuffle(copy)
    if len(copy) >= group_size:
      for j in range(0,group_size):
        line = copy.pop(0)
        line = line.split("\t")[0].lower()
        group.append(line)
    seq = get_final_sequence(group, initial_num, group_size)
    seq_list.append(seq)
    print("Found seq", seq)
  return seq_list

def get_frequent(seq_list, data):

  correspond = [0]*len(seq_list)
  total = len(data)
  len_list = []
  trimed_list = []
  for x in seq_list:
    len_list.append(len(x))
  for line in data:
    line = line.split("\t")[0].lower()
    iter_seq_list = []
    for seq in seq_list:
      iter_seq = (x for x in seq)
      iter_seq_list.append(iter_seq)
    for i in range(len(iter_seq_list)):
      if match(line, iter_seq_list[i], len_list[i]):
        correspond[i] += 1
      else:
        continue
  for i in range(len(correspond)):
    seq = seq_list[i]
    if float(correspond[i])/total <= .95:
      print("Discarded seq", seq)
    else:
      trimed_list.append(seq)

  return set(trimed_list)



def sequence_mining(data, sequence, start, end):
  for i in range(start, end):
    sequence = get_common(sequence, data[i])
  return sequence

def match(string, iter_seq, seq_len, discontinue = 0):
  discontinues = discontinue
  index = 0
  limit = 0.5 * seq_len
  string = string.lower()
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # !!!the discontinue parameter!!!
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  correspond = 1
  try:
    character = next(iter_seq)
    if character in string:
      index = string.index(character)
      if index > 0:
        discontinues += 1
        if discontinues >= limit:
          correspond = 0
          return correspond
      correspond = match(string[index+1:], iter_seq, seq_len, discontinues)
      # print(correspond, discontinues)
    else:
      correspond = 0
    return correspond
# Not correspond if a char in seq is not in data
  except StopIteration as e:
    return correspond
# The items in sequence must appear in order
# So using recurrence to match the string


def get_final_sequence(data, initial_num, end_mining):
  init_sequence = base_sequence(data, initial_num)
  final_sequence = sequence_mining(data, init_sequence, initial_num + 1, end_mining)
  return final_sequence


def list_match(string, seq_list):
  correspond = 0
  matched = 0
  for seq in seq_list:
    seq_len = len(seq)
    seq = (x for x in seq)
    if match(string, seq, seq_len):
      correspond += 1
    else:
      continue
  if correspond > len(seq_list)/2.0:
    matched = 1
  else:
    matched = 0

  return matched




# if __name__ == "__main__":
#   data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/"
#   filename = "iOS 6.1.txt"
#   headers = get_lines(data_dir, filename)
#   initial_num = 10
#   end_mining = 1000
#   correspond = 0

#   base_sequence = base_sequence(headers, initial_num, 5)
#   print(base_sequence)
#   final_sequence = sequence_mining(headers, base_sequence, initial_num + 1, end_mining)
#   final_sequence = "".join(final_sequence[1])

#   for head in headers[end_mining:]:
#     head = head.split("\t")[0]
#     iter_seq = (x for x in final_sequence)
#     if match(head, iter_seq):
#       correspond += 1

#   print(correspond)
#   print(len(headers))








