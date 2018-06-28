# -*- coding:utf-8 -*-
#! python2

__author__ = 'yoyo'

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import threading
import gzip
import pickle
import logging
import os
import os.path
import time

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def createlogger():
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)  # Log等级总开关
  # 第二步，创建一个handler，用于写入日志文件
  rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
  new_dir = './Logs/' + os.path.basename(os.sys.argv[0]).split(".")[0] + '/'
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

def to_one_hot(labels, size = 10):
  one_hot_labels = []
  for label in labels:
    oh_label = [0] * size
    oh_label[(label-1)] = 1
    one_hot_labels.append(oh_label)
  return one_hot_labels

def load_data(filename):
  with gzip.open(filename) as f:
    training_data, valid_data, test_data = pickle.load(f)
  return training_data, valid_data, test_data

class Net(nn.Module):
  """docstring for Net"""
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 3, 5)
    self.conv2 = nn.Conv2d(3, 6, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(6*5*5, 50)
    self.fc2 = nn.Linear(50, 35)
    self.fc3 = nn.Linear(35, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 6 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
# define the net

def toDataLoader(data):
  train_data, train_label = data
  train_data = train_data.reshape(len(train_data), 1, 28, 28)
  train_data = torch.from_numpy(train_data)
  train_label = torch.from_numpy(train_label)
  train_data = torch.utils.data.TensorDataset(train_data, train_label)
  train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = 4, shuffle = True, num_workers = 2)
  return train_data_loader

if __name__ == '__main__':
  logger = createlogger()

  net = Net()

  train, _ , test = load_data('./mnist.pkl.gz')
  train_data_loader = toDataLoader(train)
  test_data_loader = toDataLoader(test)

  # train_iter = iter(train_data_loader)
  # test_iter = iter(test_data_loader)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

  for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
      inputs, labels = data
      optimizer.zero_grad()

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
        logger.info('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  logger.info('Finished Training')

  # dataiter = iter(test_data_loader)
  # images, labels = dataiter.next()

  # logger.info('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
  # outputs = net(images)
  # _, predicted = torch.max(outputs,1)

  # logger.info('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
  #                             for j in range(4)))

  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_data_loader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  logger.info('Accuracy in total : %d %%' % (
    100 * correct / total))
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  with torch.no_grad():
    for data in test_data_loader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

  for i in range(10):
    logger.info('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))





