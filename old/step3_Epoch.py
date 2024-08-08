# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:37:40 2024

@author: ZKHY
"""
#  Epoch
import mne,os
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Fangsong']
plt.rcParams['axes.unicode_minus'] = False
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
# 数据导入
wd = r'D:\zkhy\jl_hmcx_zsqy\\'
os.chdir(wd)
result_save = '3result\\'
dt_load = os.listdir(result_save)
Raw = Loadpkl(result_save, dt_load[3])
file_dt = [k for k in Raw.keys()]
sfreq = 250
# 提取epoch
duration = 2
tmin=0
tmax=duration
description=['R']
Epochs = {}   
fn = file_dt[17]
for fn in file_dt:
    # 提取epoch
    raw = Raw[fn]
    n_samples = len(raw)  # 总采样点
    n_seconds = n_samples / sfreq   # 总秒数
    onset = np.arange(0, n_seconds, duration)  # 生成标记时间点（秒）
    # 创建注释对象Built-in muta
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    # 将注释添加到 Raw 对象中
    Raw_mark = raw.copy().set_annotations(annotations)
    
    events,event_dict = mne.events_from_annotations(Raw_mark)
    Epochs[fn] = mne.Epochs(Raw_mark, events=events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True,
                        baseline=(0, 0))
# 删除用不了的数据
del Epochs['444_zsqy_waterEEG.edf']

# 删除坏段
Epo_DroPA = {}
reject = dict(eeg=1.0e-4) # 删除坏段的标准 PTP最大峰值差
reject_PF = 7.5e-5 # 删除坏段的标准 PF峰值频率
# 删除坏段和坏导
Epo_DroPA = {}
reject = dict(eeg=2.0e-4) # 删除坏段的标准 PTP最大峰值差
reject_PF = 1.0e-4 # 删除坏段的标准 PF峰值频率
for fn in Epochs.keys():
    epoch = Epochs[fn].copy()
    # 删除最大峰值差大于阈值的坏段 
    Epo_DroPA[fn] = epoch.copy().drop_bad(reject=reject)
    # 删除绝对值超过PF的段
    abs_reject_indices = [i for i, e in enumerate(Epo_DroPA[fn].get_data()) if (np.abs(e) > reject_PF).any()]
    Epo_DroPA[fn].drop(abs_reject_indices, reason='ABS_AMP')

# 坏导处理
fn = '姜雷20240223刺激后_EEG.edf'
Epoch_bad = {k:Epo_DroPA[k] for k in Epo_DroPA.keys() if len(Epo_DroPA[k])<20} 
ch_bad = pd.read_excel(result_save+'分析记录.xlsx',sheet_name=1)
Epo_DroPA2 = {}
for fn in Epoch_bad.keys():
    epoch = Epochs[fn].copy()
    ch = [ch_bad['坏导'][i] for i in range(ch_bad.shape[0]) if ch_bad['记录次数'][i][0:-4]==fn[0:-4]]
    print(ch)
    if ch[0]==ch[0]:
        if len(eval(ch[0])) == 1:
            epoch.info['bads'].append(eval(ch[0])[0])  
        else:
            epoch.info['bads'].extend(eval(ch[0])) 
    #epoch.interpolate_bads() # 插值坏导
    epoch.drop_channels(epoch.info['bads']) # 删除坏导
    Epo_DroPA2[fn] = epoch.copy().drop_bad(reject=reject)
    abs_reject_indices = [i for i, e in enumerate(Epo_DroPA2[fn].get_data()) if (np.abs(e) > reject_PF).any()]
    Epo_DroPA2[fn].drop(abs_reject_indices, reason='ABS_AMP')

Epo_DroPA.update(Epo_DroPA2)
Epoch_use = {k:Epo_DroPA[k] for k in Epo_DroPA.keys() if len(Epo_DroPA[k])>0} 
Savepkl(result_save,'step3_EpochUsefulV2.pkl',Epoch_use)  

for fn in Epoch_use.keys():
    #dt = Epoch_use[fn].copy()
    Epoch_use[fn].plot()
    #Epoch_bad[fn].plot_drop_log()
    plt.title(fn)
   
   