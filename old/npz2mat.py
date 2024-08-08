# 本脚本的主要功能是将npz格式的数据，转换成mat格式的数据并保存

import numpy as np
from scipy.io import savemat

# 输入需要转换的文件的路径
data = np.load(r"E:\华意的相关资料\00-医院的数据\一些测试的数据\海绵盐水电极 Fp1 Fp2_EEG.npz")

# 转换后，进行保存输出的路径
mat_file_path = r"E:\华意的相关资料\00-医院的数据\一些测试的数据\海绵盐水电极 Fp1 Fp2_EEG.mat"

rawdata = data['eeg_raw_data']
channel = data["channel_names"]
rawdata = np.array(rawdata)

# 进行数据转换，将数据从三维矩阵转变成二维矩阵（通道数，时间样本点数）
single_data = rawdata[0, :, :]
new_data = np.zeros([single_data.shape[0], 10])
for i in range(rawdata.shape[0]):
    print(i)
    single_data = rawdata[i, :, :]
    new_data = np.concatenate([new_data, single_data], axis=1)
new_data = new_data[:, 10:]

# 使用 savemat 函数将数组保存为 .mat 文件
data_array = new_data
savemat(mat_file_path, {'data': data_array})

print('一共采集了', np.array(data_array).shape[1]/250/60, '分钟')
print('一共采集了', np.size(channel), '通道，采集的通道名称为：', channel)
print('\n转换完成~~~')




