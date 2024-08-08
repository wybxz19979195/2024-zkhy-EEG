# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:53:42 2024

@author: ZKHY
"""
import mne,os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, lfilter
import scipy.signal as signal
from mne.preprocessing import ICA
def Loadpkl(file, filename):
    # load file.pickle. file:file's name
    file = file + filename
    if os.path.exists(file):
        with open(file, 'rb') as file:
            data = pkl.load(file)
            return data
    else:
        print(file)
def Savepkl(file, filename, data):
    # save file as .pickle. file:file's name
    filesave = file + filename
    if os.path.exists(filesave):
        print(filesave)
    else:
        file = open(filesave, 'wb')
        pkl.dump(data, file)
        file.close()
def get_bandstop_filter_para(order, cutoff_freq, fs):
    nyquist_freq = fs / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    filter_order = order
    b, a = signal.butter(filter_order, normalized_cutoff, btype='bandstop')
    zi = signal.lfilter_zi(b, a)
    return b, a, zi
def get_bandpass_filter_para(order, cutoff_freq, fs):
    nyquist_freq = fs / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    filter_order = order
    b, a = signal.butter(filter_order, normalized_cutoff, btype='bandpass')
    zi = signal.lfilter_zi(b, a)
    return b, a, zi
def filt_process(indata, num_channels, num_samples, notch50_b, notch50_a, notch100_b, notch100_a, pass_b, pass_a):
    filt_data = np.zeros((num_channels, num_samples))
    for channel in range(num_channels):
        det_data = signal.detrend(indata[channel, :])
        notch50_filter = signal.lfilter(notch50_b, notch50_a, det_data)
        notch100_filter = signal.lfilter(notch100_b, notch100_a, notch50_filter)
        pass_filter = signal.lfilter(pass_b, pass_a, notch100_filter)
        filt_data[channel, :] = pass_filter
    return filt_data

# 数据导入
wd = r'D:\zkhy\jl_hmcx_zsqy\\'
os.chdir(wd)
dt_load = '1data\\' #瑞金数据\\
result_save = '3result\\'
file_dt = os.listdir(dt_load)
montage = mne.channels.make_standard_montage('standard_1020')
# %%预处理
# 获取滤波系数
sfreq = 250
notch50_b, notch50_a, notch50_zi = get_bandstop_filter_para(order=4, cutoff_freq=np.array([48, 52]), fs=sfreq)
notch100_b, notch100_a, notch100_zi = get_bandstop_filter_para(order=4, cutoff_freq=np.array([98, 102]), fs=sfreq)
pass_b, pass_a, pass_zi = get_bandpass_filter_para(order=4, cutoff_freq=np.array([0.5, 60]), fs=sfreq)
raw_FilRef = {}
for i in range(len(file_dt)):
    if i > 0:
        fn = file_dt[i]
        print(fn)
        
        path = dt_load+fn+'\\'+'EEG_data.fif'
        # 读取文件
        raw = mne.io.read_raw_fif(path,preload=True)
        raw._data = raw._data/10e6 # 单位转换uv--v
        raw.set_montage(montage)
        # 滤波 
        pre_data = raw._data.copy()
        pre_num_channels, pre_num_samples = pre_data.shape
        pre_filt_data = filt_process(pre_data, pre_num_channels, pre_num_samples,
                                         notch50_b, notch50_a, notch100_b, notch100_a, pass_b, pass_a)
        raw_Fil = raw.copy()
        raw_Fil._data = pre_filt_data
        # 重参考
        raw_FilRef[fn] = raw_Fil.copy()#.set_eeg_reference('average', projection=True)
dt_old = os.listdir(dt_load+'//old')

for fn in dt_old:
    path = dt_load +'//old//'+ fn
    print(fn)
    raw = mne.io.read_raw_edf(path,preload=True)
    raw.set_montage(montage)
    # 滤波 
    if '20240223' in fn or '20240225' in fn:
        print(fn)
        pre_data = raw._data.copy()*223.5174
    else: 
        pre_data = raw._data.copy()
    pre_num_channels, pre_num_samples = pre_data.shape
    pre_filt_data = filt_process(pre_data, pre_num_channels, pre_num_samples,
                                 notch50_b, notch50_a, notch100_b, notch100_a, pass_b, pass_a)
    raw_Fil = raw.copy()
    raw_Fil._data = pre_filt_data
   
    raw_FilRef[fn] = raw_Fil.copy()

Savepkl(result_save, 'step1_EEGFilter.pkl', raw_FilRef)


