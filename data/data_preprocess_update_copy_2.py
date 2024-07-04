import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from utils.preprocess import get_ecg,get_img,get_tar,get_cha,get_I,get_snp



test_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/test_v5.csv',dtype={176: str})
test_cmr = test_csv['20209_2_0'].values
test_eid = test_csv['eid'].values

print(test_eid.shape)
print(type(test_eid))


def print_and_log(message, file):
    print(message)
    print(message, file=file)
    file.flush()
    os.fsync(file.fileno())


with open('data_updata_output_test.txt', 'w') as f:
  
    test_cmr_list = []
    for i, data in enumerate(test_cmr):
        print_and_log(f'processing test_cmr_data {i} ...', f)
        test_cmr_list.append(get_img(data, is_continuous=True))

    test_pt = torch.load('/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/test_data_dict_v5.pt')
    print('loading data dict ......')
    if 'test_cmr_data' in test_pt:
        del test_pt['test_cmr_data']
    test_pt['test_cmr_data'] = torch.from_numpy(np.array(test_cmr_list)).float()
    test_pt['test_eid'] = torch.from_numpy(test_eid)

torch.save(test_pt,'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/test_data_dict_v6.pt')

