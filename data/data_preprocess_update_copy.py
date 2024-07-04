import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from utils.preprocess import get_ecg,get_img,get_tar,get_cha,get_I,get_snp



val_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v5.csv',dtype={176: str})
val_cmr = val_csv['20209_2_0'].values
val_eid = val_csv['eid'].values

print(val_eid.shape)
print(type(val_eid))


def print_and_log(message, file):
    print(message)
    print(message, file=file)
    file.flush()
    os.fsync(file.fileno())


with open('data_updata_output_val.txt', 'w') as f:
  
    val_cmr_list = []
    for i, data in enumerate(val_cmr):
        print_and_log(f'processing val_cmr_data {i} ...', f)
        val_cmr_list.append(get_img(data, is_continuous=True))

    val_pt = torch.load('/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt')
    print('loading data dict ......')
    if 'val_cmr_data' in val_pt:
        del val_pt['val_cmr_data']
    val_pt['val_cmr_data'] = torch.from_numpy(np.array(val_cmr_list)).float()
    val_pt['val_eid'] = torch.from_numpy(val_eid)

torch.save(val_pt,'/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v6.pt')

