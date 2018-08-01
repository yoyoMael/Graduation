#! python3
# -*- coding:utf-8 -*-

# Useful discriminator top 11:
# 1.port
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
# Columns (68,69,72,73,98,99,100,101,109,110,111,208,224,225,226,227,234,235,236,237,242,243,244,245,246,247) have mixed types.

import numpy as np
import pandas as pd
import itertools
import os
import time
import xgboost as xgb
import operator
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB as gnb
from moore_preprocessing import to_str


def get_data(df, index):
    # uses predefined index
    data = df[index].values
    # get the data from assigned index, index must be single or in a list
    labels = df['248'].values
    # df.values or np.array(df) are OK
    return data, labels
# get the data by index


def generate_train_df(names):
    df_list = []
    for suffix in range(10):
      filename = prefix + to_str(suffix)
      file_dir = os.path.join(data_dir, filename)
      if os.path.exists(file_dir):
        print(file_dir)
        df = pd.read_csv(file_dir, names = names)
        df_list.append(df)
    data = pd.concat(df_list)
    return data
# read the first 9 files, use it as train data

def split(arr):
    return arr[:20000], arr[20000:]
# split array

def fit_clf(data, labels, clf):
    clf.fit(data, labels)
    return clf


def get_score(test_data, test_labels, clf):
    score = clf.score(test_data, test_labels)
    return score

def train_step(df, index, clf):
    train_data, train_labels = get_data(df, index)
    clf.partial_fit(train_data, train_labels, classes)
    return clf

# one step of train, using partial_fit

def get_str_index(index):
    str_index = []
    for x in index:
        str_index.append(str(x))
    return np.array(str_index)

# def plus_n_minus_r(n, r, index, full_index):
#   if n > r:
#     candidate = np.array([])
#     random_index = index.copy()
#     index_len = len(random_index) + n - r

#     for i in full_index:
#       if i in index:
#         continue
#       else:
#         candidate = np.append(candidate, i)

#     if len(candidate) > n:
#       candidate = np.random.choice(candidate, n)

#     random_index = np.r_[random_index,candidate]
#     random_index = np.random.choice(random_index,index_len)

#   return random_index

def plus_n_minus_r(index, full_index, n, r):
    if n > r:
        index_len = len(index) + n - r
        index_list = []
        candidate = []
        add_list = []

        for i in full_index:
          if i in index:
            continue
          else:
            candidate.append(i)

        add_list = [list(x) for x in itertools.combinations(candidate, n)]

        candidate = []

        for add in add_list:
          add.extend(index)
          candidate.append(add)

        for c in candidate:
          add_list =  [list(x) for x in itertools.combinations(c, index_len)]
          for add in add_list:
            if add not in index_list:
              index_list.append(add)

    return index_list


def label_to_num(label_seq, classes):
    label_seq = [classes.index(x) for x in label_seq]
    label_seq = np.array(label_seq)
    return label_seq
# xgboost requires number label, NN requires one-hot coded label
plt.barh
data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore'
# full_index = np.array(['0', '59', '94', '95', '84', '161', '44', '179', '82', '112', '58'])
non_num = [68,69,72,73,98,99,100,101,109,110,111,208,224,225,226,227,234,235,236,237,242,243,244,245,246,247,248]
full_index = [x for x in range(249) if x not in non_num]
  # num_index = np.array([0, 59, 94, 95, 84, 161, 44, 179, 82, 112, 58])
  # currently, using the first four is the most accurate
prefix = 'entry'
classes = ['WWW','MAIL','FTP-CONTROL','FTP-PASV','ATTACK','P2P','DATABASE','FTP-DATA','MULTIMEDIA','SERVICES','INTERACTIVE','GAMES']
# used to tell gnb.partial_fit  all the possible classes in labels

def create_feature_map(features):
    outfile = open(os.path.join(data_dir,'xgb.fmap'), 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
        i = i + 1
    outfile.close()
        # str.format is faster than str%(a,b,c), but str1+a+b+c is faster, but its usage is limited
if __name__ == '__main__':
    names = [x for x in range(249)]
    train_df = generate_train_df(names)
    test_dir = os.path.join(data_dir, 'entry10')
    test_df = pd.read_csv(test_dir, names = names)
    full_index = [x for x in full_index if type(test_df[x].values[0]) is not str]
    # test_data = test_df[['0','82']].values
    train_labels = train_df[248].values
    train_labels = label_to_num(train_labels, classes)
    test_labels = test_df[248].values
    test_labels = label_to_num(test_labels, classes)

    train_data = train_df[full_index].values
    test_data = test_df[full_index].values

    dtrain = xgb.DMatrix(train_data, label = train_labels)
    dtest = xgb.DMatrix(test_data, label = test_labels)

    clf = xgb.train({},dtrain,20)
    features = [x for x in test_df[full_index].columns]
    create_feature_map(features)

    importance = clf.get_fscore(fmap=os.path.join(data_dir,'xgb.fmap'))
    # remember the format of the feature map!!!
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv(os.path.join(data_dir, 'featuer_importance.csv'), index = False)

    # plt.figure()
    # df.plot(kind='bar', x='feature', y='fscore', legend=False, figsize=(27,12))
    # plt.title('XGBoost Feature Importance')
    # plt.xlabel('realavive importance')
    # plt.savefig(os.path.join(data_dir,'Featureimportance.png'))
    # plt.show()
