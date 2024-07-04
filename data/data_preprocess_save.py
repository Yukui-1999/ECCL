import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from utils.preprocess import get_ecg,get_img,get_tar,get_cha,get_I,get_snp

train_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/train_v5.csv',dtype={176: str})
val_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v5.csv',dtype={176: str})
test_csv = pd.read_csv('/mnt/data/ukb_collation/ukb_ecg_cmr/data/test_v5.csv',dtype={176: str})

train_ecg = train_csv['20205_2_0'].values
val_ecg = val_csv['20205_2_0'].values
test_ecg = test_csv['20205_2_0'].values

train_cmr = train_csv['20209_2_0'].values
val_cmr = val_csv['20209_2_0'].values
test_cmr = test_csv['20209_2_0'].values

train_tar , val_tar , test_tar = get_tar(train_csv,val_csv,test_csv)
train_snp , val_snp , test_snp = get_snp(train_csv,val_csv,test_csv)
train_cha,val_char,test_char = get_cha(train_csv,val_csv,test_csv)
train_I21,val_I21,test_I21 = get_I('I21',train_csv,val_csv,test_csv)
train_I42,val_I42,test_I42 = get_I('I42',train_csv,val_csv,test_csv)
train_I48,val_I48,test_I48 = get_I('I48',train_csv,val_csv,test_csv)
train_I50,val_I50,test_I50 = get_I('I50',train_csv,val_csv,test_csv)

# print(train_ecg.shape)
# print(type(train_ecg))
# print(train_tar.info())
train_ecg_data = []
train_cmr_data = []
train_tar_data= []
train_snp_data = []
train_cha_data = []
train_I21_data = []
train_I42_data = []
train_I48_data = []
train_I50_data = []

val_ecg_data = []
val_cmr_data = []
val_tar_data = []
val_snp_data = []
val_cha_data = []
val_I21_data = []
val_I42_data = []
val_I48_data = []
val_I50_data = []

test_ecg_data = []
test_cmr_data = []
test_tar_data = []
test_snp_data = []
test_cha_data = []
test_I21_data = []
test_I42_data = []
test_I48_data = []
test_I50_data = []



for i in range(val_ecg.shape[0]):
    print(f'val processing {i}th data')

    ecg_path = val_ecg[i]
    ecg_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20205', ecg_path)
    ecg = get_ecg(ecg_path)
    # ecg = torch.from_numpy(ecg).float()
    val_ecg_data.append(ecg)

    cmr_path = val_cmr[i]
    img = get_img(cmr_path,is_continuous=True)
    # img = torch.from_numpy(img).float()
    val_cmr_data.append(img)


    tar = val_tar.iloc[i]
    # tar = torch.from_numpy(np.array(tar)).float()
    val_tar_data.append(tar)

    snp = val_snp.iloc[i]
    val_snp_data.append(snp)

    cha = val_char.iloc[i]
    val_cha_data.append(cha)

    I21 = val_I21.iloc[i]
    val_I21_data.append(I21)

    I42 = val_I42.iloc[i]
    val_I42_data.append(I42)

    I48 = val_I48.iloc[i]
    val_I48_data.append(I48)

    I50 = val_I50.iloc[i]
    val_I50_data.append(I50)


val_ecg_data = torch.from_numpy(np.array(val_ecg_data)).float()
val_cmr_data = torch.from_numpy(np.array(val_cmr_data)).float()
val_tar_data = torch.from_numpy(np.array(val_tar_data)).float()
val_snp_data = torch.from_numpy(np.array(val_snp_data)).float()
val_cha_data = torch.from_numpy(np.array(val_cha_data)).float()
val_I21_data = torch.from_numpy(np.array(val_I21_data)).float()
val_I42_data = torch.from_numpy(np.array(val_I42_data)).float()
val_I48_data = torch.from_numpy(np.array(val_I48_data)).float()
val_I50_data = torch.from_numpy(np.array(val_I50_data)).float()

print(f'val_ecg_data.shape: {val_ecg_data.shape}, val_cmr_data.shape: {val_cmr_data.shape}, val_tar_data.shape: {val_tar_data.shape},val_snp_data.shape : {val_snp_data.shape}, val_cha_data.shape: {val_cha_data.shape}, val_I21_data.shape: {val_I21_data.shape}, val_I42_data.shape: {val_I42_data.shape}, val_I48_data.shape: {val_I48_data.shape}, val_I50_data.shape: {val_I50_data.shape}')
val_data_dict = {
    'val_ecg_data': val_ecg_data,
    'val_cmr_data': val_cmr_data,
    'val_tar_data': val_tar_data,
    'val_snp_data': val_snp_data,
    'val_cha_data': val_cha_data,
    'val_I21_data': val_I21_data,
    'val_I42_data': val_I42_data,
    'val_I48_data': val_I48_data,
    'val_I50_data': val_I50_data,
}
torch.save(val_data_dict, '/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt')




for i in range(train_ecg.shape[0]):
    print(f'train processing {i}th data')
    ecg_path = train_ecg[i]
    ecg_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20205', ecg_path)
    ecg = get_ecg(ecg_path)
    # ecg = torch.from_numpy(ecg).float()
    train_ecg_data.append(ecg)

    cmr_path = train_cmr[i]
    img = get_img(cmr_path)
    # img = torch.from_numpy(img).float()
    train_cmr_data.append(img)


    tar = train_tar.iloc[i]
    # tar = torch.from_numpy(np.array(tar)).float()
    train_tar_data.append(tar)

    snp = train_snp.iloc[i]
    train_snp_data.append(snp)

    cha = train_cha.iloc[i]
    train_cha_data.append(cha)

    I21 = train_I21.iloc[i]
    train_I21_data.append(I21)

    I42 = train_I42.iloc[i]
    train_I42_data.append(I42)

    I48 = train_I48.iloc[i]
    train_I48_data.append(I48)

    I50 = train_I50.iloc[i]
    train_I50_data.append(I50)


train_ecg_data = torch.from_numpy(np.array(train_ecg_data)).float()
train_cmr_data = torch.from_numpy(np.array(train_cmr_data)).float()
train_tar_data = torch.from_numpy(np.array(train_tar_data)).float()
train_snp_data = torch.from_numpy(np.array(train_snp_data)).float()
train_cha_data = torch.from_numpy(np.array(train_cha_data)).float()
train_I21_data = torch.from_numpy(np.array(train_I21_data)).float()
train_I42_data = torch.from_numpy(np.array(train_I42_data)).float()
train_I48_data = torch.from_numpy(np.array(train_I48_data)).float()
train_I50_data = torch.from_numpy(np.array(train_I50_data)).float()
print(f'train_ecg_data.shape: {train_ecg_data.shape}, train_cmr_data.shape: {train_cmr_data.shape}, train_tar_data.shape: {train_tar_data.shape}, train_snp_data.shape : {train_snp_data.shape}, train_cha_data.shape: {train_cha_data.shape}, train_I21_data.shape: {train_I21_data.shape}, train_I42_data.shape: {train_I42_data.shape}, train_I48_data.shape: {train_I48_data.shape}, train_I50_data.shape: {train_I50_data.shape}')
train_data_dict = {
    'train_ecg_data': train_ecg_data,
    'train_cmr_data': train_cmr_data,
    'train_tar_data': train_tar_data,
    'train_snp_data': train_snp_data,
    'train_cha_data': train_cha_data,
    'train_I21_data': train_I21_data,
    'train_I42_data': train_I42_data,
    'train_I48_data': train_I48_data,
    'train_I50_data': train_I50_data,
}
torch.save(train_data_dict, '/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/train_data_dict_v5.pt')

for i in range(test_ecg.shape[0]):
    print(f'test processing {i}th data')
    ecg_path = test_ecg[i]
    ecg_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20205', ecg_path)
    ecg = get_ecg(ecg_path)
    # ecg = torch.from_numpy(ecg).float()
    test_ecg_data.append(ecg)

    cmr_path = test_cmr[i]
    img = get_img(cmr_path)
    # img = torch.from_numpy(img).float()
    test_cmr_data.append(img)


    tar = test_tar.iloc[i]
    # tar = torch.from_numpy(np.array(tar)).float()
    test_tar_data.append(tar)

    snp = test_snp.iloc[i]
    test_snp_data.append(snp)

    cha = test_char.iloc[i]
    test_cha_data.append(cha)

    I21 = test_I21.iloc[i]
    test_I21_data.append(I21)

    I42 = test_I42.iloc[i]
    test_I42_data.append(I42)

    I48 = test_I48.iloc[i]
    test_I48_data.append(I48)

    I50 = test_I50.iloc[i]
    test_I50_data.append(I50)

test_ecg_data = torch.from_numpy(np.array(test_ecg_data)).float()
test_cmr_data = torch.from_numpy(np.array(test_cmr_data)).float()
test_tar_data = torch.from_numpy(np.array(test_tar_data)).float()
test_snp_data = torch.from_numpy(np.array(test_snp_data)).float()
test_cha_data = torch.from_numpy(np.array(test_cha_data)).float()
test_I21_data = torch.from_numpy(np.array(test_I21_data)).float()
test_I42_data = torch.from_numpy(np.array(test_I42_data)).float()
test_I48_data = torch.from_numpy(np.array(test_I48_data)).float()
test_I50_data = torch.from_numpy(np.array(test_I50_data)).float()
print(f'test_ecg_data.shape: {test_ecg_data.shape}, test_cmr_data.shape: {test_cmr_data.shape}, test_tar_data.shape: {test_tar_data.shape},test_snp_data.shape: {test_snp_data.shape}, test_cha_data.shape: {test_cha_data.shape}, test_I21_data.shape: {test_I21_data.shape}, test_I42_data.shape: {test_I42_data.shape}, test_I48_data.shape: {test_I48_data.shape}, test_I50_data.shape: {test_I50_data.shape}')
test_data_dict = {
    'test_ecg_data': test_ecg_data,
    'test_cmr_data': test_cmr_data,
    'test_tar_data': test_tar_data,
    'test_snp_data': test_snp_data,
    'test_cha_data': test_cha_data,
    'test_I21_data': test_I21_data,
    'test_I42_data': test_I42_data,
    'test_I48_data': test_I48_data,
    'test_I50_data': test_I50_data,
}
torch.save(test_data_dict, '/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/test_data_dict_v5.pt')
