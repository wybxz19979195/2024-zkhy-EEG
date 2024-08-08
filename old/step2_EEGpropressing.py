# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:43:08 2024

@author: ZKHY
"""
import mne,os
import numpy as np
import pandas as pd
import pickle as pkl
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

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
wd = 'D:\\zkhy\\jl_hmcx_zsqy\\'
os.chdir(wd)
dt_load = '1rawdata\\'
result_save = '3result\\'
file_dt = os.listdir(dt_load)
fre_h, fre_l = 1,120
sfreq = 250
montage = mne.channels.make_standard_montage('standard_1020')
raw_FilCh = {}

# %%
for fn in  file_dt:
#fn = '姜雷20240223刺激后_EEG.pkl'
    raw = Loadpkl(dt_load,fn)
    raw.set_montage(montage)
    # 对数据进行滤波
    notch_freqs = [ 50,  100] # 公频倍频
    raw_Fil = raw.copy().notch_filter(freqs=notch_freqs,notch_widths=1.25,fir_window='blackman')
    raw_Fil = raw_Fil.filter(fre_h, fre_l, fir_design='firwin')
    # 重参考
    raw_FilRef = raw_Fil.set_eeg_reference('average', projection=True)
    # 删坏导
    raw_FilCh[fn] = raw_Fil
    
std_threshold = 0
channel_std = raw_FilRef.get_data().std(axis=1)
bad_channels = [ch_name for ch_name, std in zip(raw_FilRef.ch_names, channel_std) if std < std_threshold]
raw_Fil.plot(scalings = 4000)
raw_FilCh[fn] = raw_Fil
raw_FilCh[fn].plot(scalings = 4000)
Savepkl(result_save,'step2_Raw-Filter-Dropbadchannel'+'.pkl',raw_FilCh)
# %%主成分分析ICA 和提取epoch
duration = 5  # 每5秒一个标记
tmin=0
tmax=duration
description=['R']
Epochs = {}
for fn in file_dt:
    raw_pre = raw_FilCh[fn]
    ica = ICA(n_components=8, random_state=97)
    ica.fit(raw_pre.copy() )
    raw_FilRefICA = ica.apply(raw_pre.copy() )
    # mark/dicide data
    n_samples = len(raw_FilRefICA)  # 总采样点
    n_seconds = n_samples / sfreq   # 总秒数
    onset = np.arange(0, n_seconds, duration)  # 生成标记时间点（秒）
    # 创建注释对象
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    # 将注释添加到 Raw 对象中
    Raw_mark = raw_FilRefICA.copy().set_annotations(annotations)
    events,event_dict = mne.events_from_annotations(Raw_mark)
    Epochs[fn] = mne.Epochs(Raw_mark, events=events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True,
                        baseline=(0, 0))

Savepkl(result_save,'step2_Epochs-'+str(tmax)+'s-'+str(fre_h)+'_'+str(fre_l)+'Hz.pkl',Epochs)

Epochs.plot(scalings = 4000)
# 绘制成分的谱图
ica.plot_sources(raw_FilRef)
# 查看成分的空间分布
ica.plot_components()
A = Loadpkl(result_save,'step2_Raw-Filter-Dropbadchannel'+'.pkl')

raw.plot_psd()
raw.plot(scalings = 4000)
raw
notch_freqs = [ 50,  100] # 公频倍频
raw_Fil = raw.copy().notch_filter(freqs=notch_freqs)
raw_Fil = raw_Fil.filter(fre_h, fre_l, fir_design='firwin')
fn = '姜雷-20240415-刺激后-静息态-5分钟_EEG.pkl'
raw_Fil.plot_psd()
