import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from utils.preprocess import get_ecg,get_img,get_tar,get_cha,get_I,get_snp

print('loading csv ......')
train_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/train_v6.csv',dtype={176: str})
val_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v6.csv',dtype={176: str})
test_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/test_v6.csv',dtype={176: str})
print('load csv done')
train_snp , val_snp , test_snp = get_snp(train_csv,val_csv,test_csv)
train_tar , val_tar , test_tar = get_tar(train_csv,val_csv,test_csv)
train_cha,val_char,test_char = get_cha(train_csv,val_csv,test_csv)



val_pt = torch.load('/home/dingzhengyao/data/ECG_CMR/val_data_dict_v6.pt')
if 'val_cha_data' in val_pt:
    del val_pt['val_cha_data']
    print('del done')
if 'val_snp_data' in val_pt:
    del val_pt['val_snp_data']
if 'val_tar_data' in val_pt:
    del val_pt['val_tar_data']

val_pt['val_cha_data'] = torch.from_numpy(val_char.values).float()
val_pt['val_snp_data'] = torch.from_numpy(val_snp.values).float()
val_pt['val_tar_data'] = torch.from_numpy(val_tar.values).float()
print('val done')


train_pt = torch.load('/home/dingzhengyao/data/ECG_CMR/train_data_dict_v6.pt')
if 'train_cha_data' in train_pt:
    del train_pt['train_cha_data']
    print('del done')
if 'train_snp_data' in train_pt:
    del train_pt['train_snp_data']
if 'train_tar_data' in train_pt:
    del train_pt['train_tar_data']

train_pt['train_cha_data'] = torch.from_numpy(train_cha.values).float()
train_pt['train_snp_data'] = torch.from_numpy(train_snp.values).float()
train_pt['train_tar_data'] = torch.from_numpy(train_tar.values).float()

print('train done')




test_pt = torch.load('/home/dingzhengyao/data/ECG_CMR/test_data_dict_v6.pt')
if 'test_cha_data' in test_pt:
    del test_pt['test_cha_data']
    print('del done')
if 'test_snp_data' in test_pt:
    del test_pt['test_snp_data']
if 'test_tar_data' in test_pt:
    del test_pt['test_tar_data']
test_pt['test_cha_data'] = torch.from_numpy(test_char.values).float()
test_pt['test_snp_data'] = torch.from_numpy(test_snp.values).float()
test_pt['test_tar_data'] = torch.from_numpy(test_tar.values).float()
print('test done')

torch.save(train_pt,'/home/dingzhengyao/data/ECG_CMR/train_data_dict_v7.pt')
torch.save(val_pt,'/home/dingzhengyao/data/ECG_CMR/val_data_dict_v7.pt')
torch.save(test_pt,'/home/dingzhengyao/data/ECG_CMR/test_data_dict_v7.pt')
