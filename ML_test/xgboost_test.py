#! python3
# -*- coding:utf-8 -*-

__author__ = 'yoyo'

import xgboost as xgb
import pandas as pd
import numpy as np
import os

full_index = np.array(['0', '59', '94', '95', '84', '161', '44', '179', '82', '112', '58'])
classes = ['WWW','MAIL','FTP-CONTROL','FTP-PASV','ATTACK','P2P','DATABASE','FTP-DATA','MULTIMEDIA','SERVICES','INTERACTIVE','GAMES']
data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore'
def create_feature_map(features):
  outfile =open('xgb.fmap', 'w')
  i = 0
  for feat in features:
    outfile.write('{0}\t{1}q\n'.format(i, feat))
    i = i+1
  outfile.close()

df_train = pd.read_csv(os.path.join(data_dir,'entry10'))
df_test = pd.read_csv(os.path.join(data_dir,'entry01'))

dtrain = df_train[full_index].values
dtest = df_test[full_index].values

label_train = df_train['248'].values
label_test = df_test['248'].values

label_train = [classes.index(x) for x in label_train]
label_test = [classes.index(x) for x in label_test]
label_train = np.array(label_train)
label_test = np.array(label_test)

dtrain = xgb.DMatrix(dtrain, label_train)
dtest = xgb.DMatrix(dtest, label_test)

bst = xgb.train({'silent':0, },dtrain,20)
preds = bst.predict(dtest)

pred = np.array([])

for x in preds:
  pred = np.append(pred, round(x))


print(np.mean(pred==label_test))