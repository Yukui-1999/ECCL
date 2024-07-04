import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from utils.preprocess import get_ecg,get_img,get_tar,get_cha,get_I,get_snp
import sys
sys.path.append('/home/dingzhengyao/Work/ECG_CMR/ECG_CMR_TAR/Project_version1')
print('loading csv ......')
train_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/train_v6.csv',dtype={176: str})
val_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v6.csv',dtype={176: str})
test_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/test_v6.csv',dtype={176: str})
print('load csv done')

train_I21,val_I21,test_I21 = get_I('I21',train_csv,val_csv,test_csv)
train_I42,val_I42,test_I42 = get_I('I42',train_csv,val_csv,test_csv)
train_I48,val_I48,test_I48 = get_I('I48',train_csv,val_csv,test_csv)
train_I50,val_I50,test_I50 = get_I('I50',train_csv,val_csv,test_csv)



val_pt = torch.load('/home/dingzhengyao/data/ECG_CMR/val_data_dict_v7.pt')
if 'val_I21_data' in val_pt and 'val_I42_data' in val_pt and 'val_I48_data' in val_pt and 'val_I50_data' in val_pt:
    del val_pt['val_I21_data']
    del val_pt['val_I42_data']
    del val_pt['val_I48_data']
    del val_pt['val_I50_data']
    print('del done')


val_pt['val_I21_data'] = torch.from_numpy(val_I21.values).float()
val_pt['val_I42_data'] = torch.from_numpy(val_I42.values).float()
val_pt['val_I48_data'] = torch.from_numpy(val_I48.values).float()
val_pt['val_I50_data'] = torch.from_numpy(val_I50.values).float()
print('val done')


train_pt = torch.load('/home/dingzhengyao/data/ECG_CMR/train_data_dict_v7.pt')
if 'train_I21_data' in train_pt and 'train_I42_data' in train_pt and 'train_I48_data' in train_pt and 'train_I50_data' in train_pt:
    del train_pt['train_I21_data']
    del train_pt['train_I42_data']
    del train_pt['train_I48_data']
    del train_pt['train_I50_data']

    print('del done')


train_pt['train_I21_data'] = torch.from_numpy(train_I21.values).float()
train_pt['train_I42_data'] = torch.from_numpy(train_I42.values).float()
train_pt['train_I48_data'] = torch.from_numpy(train_I48.values).float()
train_pt['train_I50_data'] = torch.from_numpy(train_I50.values).float()

print('train done')




test_pt = torch.load('/home/dingzhengyao/data/ECG_CMR/test_data_dict_v7.pt')
if 'test_I21_data' in test_pt and 'test_I42_data' in test_pt and 'test_I48_data' in test_pt and 'test_I50_data' in test_pt:
    del test_pt['test_I21_data']
    del test_pt['test_I42_data']
    del test_pt['test_I48_data']
    del test_pt['test_I50_data']

    print('del done')

test_pt['test_I21_data'] = torch.from_numpy(test_I21.values).float()
test_pt['test_I42_data'] = torch.from_numpy(test_I42.values).float()
test_pt['test_I48_data'] = torch.from_numpy(test_I48.values).float()
test_pt['test_I50_data'] = torch.from_numpy(test_I50.values).float()
print('test done')

torch.save(train_pt,'/home/dingzhengyao/data/ECG_CMR/train_data_dict_v7.pt')
torch.save(val_pt,'/home/dingzhengyao/data/ECG_CMR/val_data_dict_v7.pt')
torch.save(test_pt,'/home/dingzhengyao/data/ECG_CMR/test_data_dict_v7.pt')
