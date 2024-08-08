# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:36:21 2024

@author: ZKHY
"""
# 20240223 和 20240225的数据都需要将原始数据乘上223.5174
import mne,os
import pickle as pkl
import pandas as pd
from scipy.io import savemat
from numpy.fft import fft
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, lfilter
import scipy.signal as signal
from mne.time_frequency import tfr_multitaper
# 支持显示汉语
plt.rcParams['font.sans-serif'] = ['Fangsong']
plt.rcParams['axes.unicode_minus'] = False
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

# 加载数据
wd = r'D:\zkhy\jl_hmcx_zsqy\\'
os.chdir(wd)
dt_load = '1data_raw\\' #瑞金数据\\
result_save = '3result\\'
file_dt = os.listdir(dt_load)
montage = mne.channels.make_standard_montage('standard_1020')
# %%预处理
# 获取滤波系数
sfreq = 250
notch50_b, notch50_a, notch50_zi = get_bandstop_filter_para(order=4, cutoff_freq=np.array([48, 52]), fs=sfreq)
notch100_b, notch100_a, notch100_zi = get_bandstop_filter_para(order=4, cutoff_freq=np.array([98, 102]), fs=sfreq)
pass_b, pass_a, pass_zi = get_bandpass_filter_para(order=4, cutoff_freq=np.array([0.5, 60]), fs=sfreq)
# 设置mark和epoch数量
#path = r'D:\zkhy\jl_hmcx_zsqy\1data_raw\water_pb\EEG_data.fif'
#raw = mne.io.read_raw_fif(path,preload=True)
raw_FilRef = {}
fn = file_dt[8]
for fn in file_dt:
    path = dt_load + fn
    if '.pkl' in fn:
        print(fn)
        raw = Loadpkl(dt_load,fn)
    if '.edf' in fn:
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
    # 重参考
    raw_FilRef[fn] = raw_Fil.set_eeg_reference('average', projection=True)
Savepkl(result_save,'step2_EEGPropressFilterRefV2.pkl',raw_FilRef)

raw_FilRef = Loadpkl(result_save,'step2_EEGPropressFilterRef.pkl')
# %% 绘图
raw_Fil.plot()
bands = {'Alpha (8-12 Hz)': (8, 12)}
k1 = file_dt[5]
k2 = file_dt[2]
dt_1 = Epo_DroPA[k1].copy() # 导电膏数据
dt_2 = Epo_DroPA[k2].copy() # 盐水电极数据
ch_name =  dt_1.ch_names

dt_2[100:130].plot_psd_topomap(picks=ch_name)
dt_1.plot(picks=ch_name)
plt.title(k1)
# 定义频率范围和时间窗口
frequencies = np.arange(1, 50, 1)  # 1到50 Hz
n_cycles = frequencies / 2.  # 不同频率下的周期数
# 计算时频表示（Time-Frequency Representation, TFR）
power = tfr_multitaper(dt_1[0:30],n_cycles=n_cycles,freqs=frequencies)
power[0].plot(picks=['Fp1'])
