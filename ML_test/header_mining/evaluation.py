#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

from prefixspan import PrefixSpan as PS
from header_mining import *
from get_by_label import *

import os.path as path

def evaluation(dataset, sequence_str, seq_len, label, label_num):

  correspond = 0
  truth = 0
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  label = label.lower()

  for data in dataset:
    is_match = 0
    is_type = 0
    data = data.split("\t")
    labels = data[1].split("|")
    if len(labels) < label_num:
      continue
    else:
      true_label = labels[label_num-1].strip("\n").lower()
    if label in true_label:
      is_type = 1
      truth += 1
    else:
      is_type = 0
    data = data[0].lower()
    iter_seq = (x for x in sequence_str)
    if match(data, iter_seq, seq_len):
      is_match = 1
      correspond += 1
    else:
      is_match = 0
    if is_type == 1:
      if is_match == 1:
        TP += 1
      else:
        FN += 1
    else:
      if is_match == 1:
        FP += 1
      else:
        TN += 1
  print("correspond", correspond)
  print("truth", truth)

  TPR = float(TP)/(TP+FN)
  FPR = float(FP)/(FP+TN)
  TNR = 1-FPR




  return correspond, {"TP":TP,"FP":FP,"TN":TN,"FN":FN}, {"TPR":TPR, "FPR":FPR, "TNR":TNR}

if __name__ =='__main__':
  data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/"
  label = "Android 6.0"
  mining_file = label + ".txt"
  evaluation_file = "traindata.txt"

  data = get_lines(data_dir, mining_file)

  print("Extracting ", label, "features. \n")
  sequence_str = get_final_sequence(data, 5, 10000)
  print("Etracted > %s \n" % sequence_str)
  seq_len = len(sequence_str)
  # iter_seq = (x for x in sequence_str)
  # print(match("user-agent0q9q1arwin1qqqqq3q0q0", iter_seq, seq_len))
  dataset = get_lines(data_dir, evaluation_file)
  correspond, truth, rate= evaluation(dataset, sequence_str, seq_len, label, 2)

  print("Amount of matched data: %d \n" % correspond)
  print("Total data number: %d \n" % len(dataset))
  print("TP: %d, TN: %d, FP: %d, FN: %d \n"%(truth["TP"],truth["TN"],truth["FP"],truth["FN"]))
  print("TPR: %f, FPR: %f, TNR: %f"%(rate["TPR"],rate["FPR"],rate["TNR"]))


