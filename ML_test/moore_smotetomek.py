import numpy as np
import pandas as pd
import os
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalanceCascade
from collections import Counter
import glob
glob.glob()
data_dir = '/Users/Yoyo/Desktop/Graduation/ML_test/dataset/Moore'
entry = 'fusion_csv'
full_index = np.array([0, 59, 94, 95, 84, 161, 44, 179, 82, 112, 58, 248])
file = os.path.join(data_dir, entry)
classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES',
           'INTERACTIVE', 'GAMES']

df = pd.read_csv(file, names = full_index)
print(df.head())
data = df[full_index[:-1]].values
label = df[248].values

count = Counter(label)

count['WWW'] = 50000

print(count)

ros = RandomUnderSampler(ratio=count)
x_ros, y_ros = ros.fit_sample(data, label)
data = np.c_[x_ros, y_ros]
df = pd.DataFrame(data)
csv_path = os.path.join(data_dir, 'rus_csv')
df.to_csv(csv_path, index = False, header = False)