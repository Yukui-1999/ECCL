import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from utils.preprocess import get_ecg,get_img,get_tar,get_cha,get_I,get_snp


train_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/train_v5.csv',dtype={176: str})
# val_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v5.csv',dtype={176: str})
# test_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/test_v5.csv',dtype={176: str})

train_cmr = train_csv['20209_2_0'].values
# val_cmr = val_csv['20209_2_0'].values
# test_cmr = test_csv['20209_2_0'].values

train_eid = train_csv['eid'].values
# val_eid = val_csv['eid'].values
# test_eid = test_csv['eid'].values
print(train_eid.shape)
# print(val_eid.shape)
# print(test_eid.shape)
print(type(train_eid))
# print(type(val_eid))
# print(type(test_eid))

# val_pt = torch.load('/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt')
# test_pt = torch.load('/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/test_data_dict_v5.pt')

# if 'val_cmr_data' in val_pt:
#     del val_pt['val_cmr_data']
# if 'test_cmr_data' in test_pt:
#     del test_pt['test_cmr_data']


# val_pt['val_eid'] = val_eid
# test_pt['test_eid'] = test_eid

def print_and_log(message, file):
    print(message)
    print(message, file=file)
    file.flush()
    os.fsync(file.fileno())


with open('data_updata_output_train.txt', 'w') as f:
    train_cmr_list= []
    for i, data in enumerate(train_cmr):
        print_and_log(f'processing train_cmr_data {i} ...', f)
        train_cmr_list.append(get_img(data, is_continuous=True))


    train_pt = torch.load('/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt')
    print('loading data dict ......')
    if 'train_cmr_data' in train_pt:
        del train_pt['train_cmr_data']
    train_pt['train_cmr_data'] = torch.from_numpy(np.array(train_cmr_list)).float()
    train_pt['train_eid'] = torch.from_numpy(train_eid)
    # val_cmr_list = []
    # for i, data in enumerate(val_cmr):
    #     print_and_log(f'processing val_cmr_data {i} ...', f)
    #     val_cmr_list.append(get_img(data, is_continuous=True))
    # val_pt['val_cmr_data'] = torch.from_numpy(np.array(val_cmr_list)).float()

    # test_cmr_list = []
    # for i, data in enumerate(test_cmr):
    #     print_and_log(f'processing test_cmr_data {i} ...', f)
    #     test_cmr_list.append(get_img(data, is_continuous=True))
    # test_pt['test_cmr_data'] = torch.from_numpy(np.array(test_cmr_list)).float()

# with open('data_updata_output.txt', 'w') as f:
#     # 你的代码
#     for i in range(len(train_pt.get('train_ecg_data'))):
#         print_and_log(f'processing train_ecg_data {i} ...', f)
#         train_pt['train_cmr_data'][i] = torch.from_numpy(get_img(train_cmr[i],is_continuous=True)).float()
#     for i in range(len(val_pt.get('val_ecg_data'))):
#         print_and_log(f'processing val_ecg_data {i} ...', f)
#         val_pt['val_cmr_data'][i] = torch.from_numpy(get_img(val_cmr[i],is_continuous=True)).float()
#     for i in range(len(test_pt.get('test_ecg_data'))):
#         print_and_log(f'processing test_ecg_data {i} ...', f)
#         test_pt['test_cmr_data'][i] = torch.from_numpy(get_img(test_cmr[i],is_continuous=True)).float()

torch.save(train_pt,'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v6.pt')
# torch.save(val_pt,'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v6.pt')
# torch.save(test_pt,'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/test_data_dict_v6.pt')
