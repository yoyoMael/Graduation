#! python3
# -*- coding:utf-8 -*-

# 60. pushed_data_pkts_b a  (server > client)
# 95. initial_window_bytes_a b (client > server)
# 96. initial_window_bytes_b a (server > client)
# 85 avg_segm_size_b a (server > client)
# 162. med_data_ip_a b (client > server)
# 45. actual_data_pkts_a b (client > server)
# 180. var_data_wire_b a (server > client)
# 83. min_segm_size_a b
# 113. RTT_samples_a b
# 59. pushed_data_pkts_a b

import numpy as np
import pandas as pd
import os
import time
import warnings
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import  SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
from collections import Counter
from xgboost import XGBClassifier
import logging

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
  fh = logging.FileHandler(logfile, mode='w')
  fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
  # 第三步，定义handler的输出格式
  # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  formatter = logging.Formatter()
  fh.setFormatter(formatter)
  # 第四步，将logger添加到handler里面
  logger.addHandler(fh)
  # 日志
  return logger

def get_result(clf, clfname):
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()
    training_cost = end - start
    logger = createlogger()
    start = time.time()
    pred = clf.predict(x_test)
    end = time.time()
    pred_cost = end - start
    after_pred = clf.predict(after_data)
    label = y_test
    acc = clf.score(x_test, y_test)
    after_acc = clf.score(after_data, after_label)
    logger.info('Classifier name: %s' , clfname)
    logger.info('Training time consumption: %f , Predicting time consumption: %f \n'
                , training_cost, pred_cost)

    # np.save(os.join(data_dir, clfname), pred)
    for cls in classes:
        recall, precision, fpr, f1 = metric(pred, label, cls)
        logger.info('Metrics:')
        logger.info('Class: %s, recall %f, precision %f, fpr %f, f1 %f',cls,recall,precision,fpr,f1)
    logger.info('Accuracy: %f', acc)
    logger.info('\n *****************\n')
    for cls in classes:
        recall, precision, fpr, f1 = metric(after_pred, after_label, cls)
        logger.info('After 1 Year:')
        logger.info('Class: %s, recall %f, precision %f, fpr %f, f1 %f', cls, recall, precision, fpr, f1)
    logger.info('Accuracy: %f', after_acc)

def metric(pred, label, type):
    pr = np.where(pred == type, 1, 0)
    la = np.where(label == type, 1, 0)
    tp = np.sum((pr + la) == 2)
    fp = np.sum((pr - la) == 1)
    fn = np.sum((pr - la) == -1)
    tn = np.sum((pr + la) == 0)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr = fp / (fp + tn)
    # fnr = fn / (fn + tp)
    f1 = 2*recall*precision/(precision + recall)
    return (recall,precision,fpr,f1)

def get_data(file):
    df = pd.read_csv(file, names = full_index)
    print(df.head())
    data = df[data_index].values
    label = df[248].values
    return train_test_split(data, label, test_size=.5, random_state=42,
                                            stratify=label, shuffle=True)
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore'
filename = 'fusion_csv'
af_filename = 'entry12'
test_filename = 'fusion_csv'
full_index = np.array([0, 59, 94, 95, 84, 161, 44, 179, 82, 112, 58, 248])
names = [x for x in range(249)]
# full_index = np.array([95,94,82,59,0])
# data_index = np.array([44,179,112,59,82,58,84])
# data_index = np.array([0])

data_index = np.array([0, 59, 94, 95, 84, 161, 44, 179, 82, 112, 58])
# classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES',
#            'INTERACTIVE', 'GAMES']
classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA',
           'MULTIMEDIA', 'SERVICES', 'INTERACTIVE']

# file used to train, who generates x_train,x_test,y_train,y_test
# I also resampled the file `entry12`
file = os.path.join(data_dir, filename)
test_file = os.path.join(data_dir, test_filename)

if __name__ == '__main__':
    acc = []
    x_train, _, y_train, _ = get_data(file)
    _, x_test, _, y_test = get_data(test_file)
    np_dir = os.path.join(data_dir, 'estimators_100_150_5.txt')
    for i in range(50, 150, 5):
        clf = LGBMClassifier(n_estimators=i)
        clf.fit(x_train, y_train)
        print(clf.get_params())
        accuracy = clf.score(x_test, y_test)
        acc.append(accuracy)
    acc = np.array(acc)
    print(acc)
    np.savetxt(np_dir, acc)
import matplotlib.pyplot as plt
plt.legend