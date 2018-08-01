import re
import numpy as np
import pandas as pd
import glob
def get_string(file):
    f = open(file, 'r')
    string = f.read()
    f.close()
    first = string.split('*')[0]
    second = string.split('*')[-1]
    return first, second

def find_data(metric, string):
    reg = re.compile(metric+'.(.*?)[, \n]', re.S)
    return re.findall(reg, string)

metric_list = ['class', 'recall', 'precision', 'fpr', 'f1']
classes = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK','P2P', 'DATABASE', 'FTP-DATA','MULTIMEDIA', 'SERVICES', 'INTERACTIVE']

def to_csv(filename):
    first,second = get_string(filename)
    data = classes
    for metric in metric_list[1:]:
        data = np.c_[data,find_data(metric, first)]
    df = pd.DataFrame(data)
    df.to_csv(filename.split('.')[0]+'_1.csv', index = False, header = metric_list)
    data = classes
    for metric in metric_list[1:]:
        data = np.c_[data,find_data(metric, second)]
    df = pd.DataFrame(data)
    df.to_csv(filename.split('.')[0]+'_2.csv', index = False, header = metric_list)

if __name__ == '__main__':
    file_list = glob.glob('*.log')
    for filename in file_list:
        to_csv(filename)
