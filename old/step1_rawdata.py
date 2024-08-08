# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:43:08 2024

@author: ZKHY
"""
import mne,os
import pickle as pkl
import numpy as np
def Savepkl(file, filename, data):
    # save file as .pickle. file:file's name
    filesave = file + filename
    if os.path.exists(filesave):
        print(filesave)
    else:
        file = open(filesave, 'wb')
        pkl.dump(data, file)
        file.close()
def Loadpkl(file, filename):
    # load file.pickle. file:file's name
    file = file + filename
    if os.path.exists(file):
        with open(file, 'rb') as file:
            data = pkl.load(file)
            return data
    else:
        print(file)

wd = r'D:\zkhy\盐水海绵电极\\'
os.chdir(wd)
dt_save = '1data_raw\\'
result_save = '3result\\'
file_dt = os.listdir('1data\\')
sfreq = 250  # Hz
fn = file_dt[0]
path = '1data\\'+file_dt[1]
for fn in file_dt:
    if'EEG' in fn:
        print(fn)
        dt = np.load('1data\\'+file_dt[1])
        
        channel_name = [i[:] for i in dt['channel_names']]
        data = dt['eeg_raw_data']
        if data.shape[0]>0:# 判断是否记录数据
            single_data = data[0, :, :]
            new_data = np.zeros([single_data.shape[0], 10])
            for i in range(data.shape[0]):
                single_data = data[i, :, :]
                new_data = np.concatenate([new_data, single_data], axis=1)
            new_data = new_data[:, 10:]
            new_data.shape[1]
           
            info = mne.create_info(channel_name, sfreq, ch_types='eeg')
            raw = mne.io.RawArray(new_data, info)
            Savepkl(dt_save, fn[0:-4]+'.pkl', raw)

#raw = mne.io.read_raw_brainvision('D:\\zkhy\\1002\\1002.vhdr',preload=True)
