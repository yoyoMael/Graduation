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
import xgboost as xgb

from moore_preprocessing import to_str


def generate_train_df(names):
    df_list = []
    for suffix in range(10):
        filename = prefix + to_str(suffix)
        file_dir = os.path.join(data_dir, filename)
        if os.path.exists(file_dir):
            print(file_dir)
            df = pd.read_csv(file_dir, names=names)
            df_list.append(df)
    data = pd.concat(df_list)
    return data


def split(arr):
    return arr[:20000], arr[20000:]


# split array

def fit_clf(data, labels, clf):
    clf.fit(data, labels)
    return clf


def get_score(test_data, test_labels, clf):
    score = clf.score(test_data, test_labels)
    return score


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
            add_list = [list(x) for x in itertools.combinations(c, index_len)]
            for add in add_list:
                if add not in index_list:
                    index_list.append(add)

    return index_list


def test_step(index, train_df, train_labels, test_df, test_labels):
    train_data = train_df[index].values
    test_data = test_df[index].values
    train_data = xgb.DMatrix(train_data, label=train_labels)
    test_data = xgb.DMatrix(test_data, label=test_labels)
    # data, label = test_labels

    clf = xgb.train({'silent':1}, train_data, 20)
    preds = clf.predict(test_data)
    for i in range(len(preds)):
        preds[i] = round(preds[i])
    score = np.mean(preds == test_labels)
    print('\n')
    print('The index list is: ', index)
    print('List length: %d' % len(index))
    print('The score is: %.3f ' % score)

    return score, clf


# for a given index list(indicates feature selected), return the score of model

def label_to_num(label_seq, classes):
    label_seq = [classes.index(x) for x in label_seq]
    label_seq = np.array(label_seq)
    return label_seq


# xgboost requires number label, NN requires one-hot coded label

data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore'
# full_index = np.array(['0', '59', '94', '95', '84', '161', '44', '179', '82', '112', '58'])
full_index = np.array([59, 94, 95, 84, 161, 44, 179, 82, 112, 58])
# full_index = np.array([95,94,82,59,0])
# full_index = np.array([44,179,112,59,82,58,84])
# num_index = np.array([0, 59, 94, 95, 84, 161, 44, 179, 82, 112, 58])
# currently, using the first four is the most accurate
prefix = 'entry'
classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES',
           'INTERACTIVE', 'GAMES']
# used to tell gnb.partial_fit  all the possible classes in labels


if __name__ == '__main__':
    names = [x for x in range(249)]
    train_df = generate_train_df(names)
    test_dir = os.path.join(data_dir, 'entry10')
    test_df = pd.read_csv(test_dir, names=names)
    # test_data = test_df[['0','82']].values
    train_labels = train_df[248].values
    train_labels = label_to_num(train_labels, classes)
    test_labels = test_df[248].values
    test_labels = label_to_num(test_labels, classes)

    index = []
    high_score = 0
    best_clf = None
    n = 2
    r = 1

    while len(index) + n <= len(full_index):
        current_score = 1
        index_list = plus_n_minus_r(index, full_index, n, r)
        stop = 1
        for i in index_list:
            if len(i) > 0:
                current_score, clf = test_step(i, train_df, train_labels, test_df, test_labels)
                if current_score > high_score:
                    high_score = current_score
                    index = i
                    stop = 0
                    best_clf = clf
                else:
                    continue
        if stop:
            break
        else:
            print("-" * 30)
            print("Continue searching")
            print("Current index list: ", index)
            print("Current high score: %.3f" % high_score)
            print("-" * 30)

    print("*" * 50)
    print("Final feature index: ")
    print(index)
    print("High score: %f" % high_score)
    print("Parameters of the model: ")
    print("*" * 50)
    print('\n')

# later_df = pd.read_csv(os.path.join(data_dir,'entry12'))
# later_data = later_df[['0','82']].values
# later_labels = later_df['248']
# print(later_data)
# print(later_labels)

# print('Later test score: %f' % clf.score(later_data, later_labels))
