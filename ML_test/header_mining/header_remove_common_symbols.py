__author__ = "yoyo"

import re
import os.path as path

data_dir = "/Users/Yoyo/Desktop/Graduation/ML_test/dataset/device-detection-database/"
filename = "traindata.txt"

if __name__ == "__main__":
  filepath = path.join(data_dir, filename)
  f = open(filepath)
  raw_data = f.readlines()
  f.close()
  headers = []
  i = 0
  for data in raw_data:
    i += 1
    header = []
    data = data.split("\t")
    raw_content = data[0]
    raw_content = raw_content.replace(".", "")
    raw_content = raw_content.replace(" ", "")
    raw_content = raw_content.replace("/", "")
    header.append(raw_content)
    header.extend(data[1:])
    header = "\t".join(header)
    print(header)
    headers.append(header)
  f = open(path.join(data_dir, "no_agent.txt"),"w")
  f.write("".join(headers))
  f.close()


