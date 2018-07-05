#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

from prefixspan import PrefixSpan as PS
from header_mining import *
from get_by_label import *

import os.path as path
import random
import logging
import time
import os

def createlogger():
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)  # Log等级总开关
  # 第二步，创建一个handler，用于写入日志文件
  rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
  new_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/Logs/' + os.path.basename(os.sys.argv[0]).split(".")[0] + '/'
  if not os.path.exists(new_dir):
    os.mkdir(new_dir)
  log_path = new_dir
  log_name = log_path + rq + '.log'
  logfile = log_name
  fh = logging.FileHandler(logfile, mode='wb')
  fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
  # 第三步，定义handler的输出格式
  formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  fh.setFormatter(formatter)
  # 第四步，将logger添加到handler里面
  logger.addHandler(fh)
  # 日志
  return logger

def seq_list_evaluation(dataset, seq_list, label, label_num):

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
    if list_match(data, seq_list):
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
  label_num = 2
  mining_file = label + ".txt"
  evaluation_file = "Android 6.0.txt"
  data = get_lines(data_dir, mining_file)
  data1 = []
  for line in data:
    line = line.split("\t")[0].lower()
    data1.append(line)

  print("Extracting %s features. \n" % label)
  sequence_list = get_seq_list(data1, 5, 10, 1000)
  sequence_list = get_frequent(sequence_list, data1)
  print("Etracted \n")
  print("*"*20)
  for seq in sequence_list:
    print(seq)
  print("\n")
  print("*"*20)
  # iter_seq = (x for x in sequence_str)
  # print(match("user-agent0q9q1arwin1qqqqq3q0q0", iter_seq, seq_len))
  dataset = get_lines(data_dir, evaluation_file)
  correspond, truth, rate = seq_list_evaluation(dataset, sequence_list, label, label_num)

  print("Amount of matched data: %d \n" % correspond)
  print("Total data number: %d \n" % len(dataset))
  print("TP: %d, TN: %d, FP: %d, FN: %d \n"%(truth["TP"],truth["TN"],truth["FP"],truth["FN"]))
  print("TPR: %f, FPR: %f, TNR: %f"%(rate["TPR"],rate["FPR"],rate["TNR"]))


