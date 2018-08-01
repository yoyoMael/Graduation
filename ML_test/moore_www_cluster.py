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
from sklearn.cluster import k_means
from imblearn.combine import SMOTETomek
from collections import Counter
from xgboost import XGBClassifier
import logging

def get_center(file):
    data = get_www(file)
    data = data[:,:-1]
    center,_,_,_ = k_means(data, n_clusters=20000, n_jobs=-2)
    data = np.c_[center, np.array(['WWW']*20000)]
    return np.array(data)

def get_www(file):
    df = pd.read_csv(file, names = full_index)
    print(df.head())
    data = df[df[248]=='WWW'].values
    return data

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore'
filename = 'fusion_csv'
af_filename = 'entry12'
test_filename = 'fusion_csv'
full_index = np.array([0, 59, 94, 95, 84, 161, 44, 179, 82, 112, 58, 248])
names = [x for x in range(249)]
# full_index = np.array([95,94,82,59,0])
data_index = np.array([0,44,179,112,59,82,58,84])
# data_index = np.array([0])

# data_index = np.array([59, 94, 95, 84, 161, 44, 179, 82, 112, 58])
# classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES',
#            'INTERACTIVE', 'GAMES']
classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA',
           'MULTIMEDIA', 'SERVICES', 'INTERACTIVE']

# file used to train, who generates x_train,x_test,y_train,y_test
# I also resampled the file `entry12`
file = os.path.join(data_dir, filename)

data = get_center(file)

df = pd.DataFrame(data)
df.to_csv(os.path.join(data_dir,'www_cluster'), names=False, columns=False)