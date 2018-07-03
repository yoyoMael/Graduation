#! python3
# -*- coding:utf-8 -*-

__author__ = "yoyo"

from prefixspan import PrefixSpan as PS
import os
import os.path as path
import re
import logging
import time

data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database"
filename = "traindata.txt"

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

if __name__ == "__main__":
  filepath = path.join(data_dir, filename)
  f = open(filepath)
  headers = f.read()
  f.close()
  # get content
  headers = headers.split("\n")
  # split lines
  labels = []
  class_list = {}
  discard = 0
  num = 0
  logger = createlogger()

  for header in headers:
    num += 1
    data = header.split("\t")
    if len(data) >= 2:
      label = data[1].split("|")[0]
    else:
      discard += 1
      logger.info("%d data discarded,Current:\n >> %s"%(discard, header))
      continue
    if label not in labels:
      labels.append(label)
      class_list[label] = [header]
    if num%20000 == 0:
      print("%d type of devices, current device %s have %d datas"%(len(class_list),label,len(class_list[label])))
    if label in class_list:
      class_list[label].append(header)
    else:
      print("cao")
      break

  for key in class_list.keys():
    device_data = class_list[key]
    if(len(device_data) > 10):
      f = open(path.join(data_dir, key+".txt"),"w")
      f.write("\n".join(device_data))
      print("Finished" + key)






