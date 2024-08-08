# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:43:08 2024
# 20240223 和 20240225的数据都需要将原始数据乘上223.5174
@author: ZKHY
"""
import mne,os
import numpy as np
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
dt_load = '1rawdata\\' #瑞金数据\\
result_save = '3result\\'
file_dt = os.listdir(dt_load)
file_dt = [f for f in file_dt if len(f)>4]
fre_h, fre_l = 1,120
sfreq = 250
montage = mne.channels.make_standard_montage('standard_1020')

duration = 2  # 每几秒一个标记
tmin=0
tmax=duration
description=['R']
Epochs = {}
for fn in file_dt:
    raw = Loadpkl(dt_load,fn)
    raw.set_montage(montage)
    if '20240223' in fn or '20240225' in fn:
        print(fn)
        raw._data = raw._data.copy()*223.5174
    else: 
        raw._data = raw._data.copy()
    # 对数据进行滤波
    notch_freqs = [ 50,  100] # 公频倍频
    raw_Fil = raw.copy().notch_filter(freqs=notch_freqs,notch_widths=1.25,fir_window='blackman')
    raw_Fil = raw_Fil.filter(fre_h, fre_l, fir_design='firwin2')
    # 重参考
    raw_FilRef = raw_Fil.set_eeg_reference('average', projection=True)
    # 提取epoch
    n_samples = len(raw_FilRef)  # 总采样点
    n_seconds = n_samples / sfreq   # 总秒数
    onset = np.arange(0, n_seconds, duration)  # 生成标记时间点（秒）
    # 创建注释对象
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    # 将注释添加到 Raw 对象中
    Raw_mark = raw_FilRef.copy().set_annotations(annotations)
    
    events,event_dict = mne.events_from_annotations(Raw_mark)
    Epochs[fn] = mne.Epochs(Raw_mark, events=events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True,
                        baseline=(0, 0))  
Savepkl(result_save,'step2_Epochs-'+str(tmax)+'s-'+str(fre_h)+'_'+str(fre_l)+'Hz.pkl',Epochs)
# 手动删除坏段和坏导
reject=dict(eeg=150) # 删除坏段的标准
Epo_DroP = {}
Epo_DroA = {}
Epo_DroPA = {}
for fn in Epochs.keys():
    # 删除坏导
    epoch = Epochs[fn].copy()
    #epoch.info['bads'].extend(['P3', 'FC3' ,'Cz', 'FCz', 'PO3', 'P7', 'CPz', 'C3', 'Pz', 'P8']) 
    #epoch.info['bads'].append('P3')
    Epo_DroP[fn] = epoch.copy().drop_bad(reject=reject) # PTP
    Epo_DroA[fn] = epoch.copy().drop_bad(reject=dict(eeg=lambda x: ((np.abs(x) >75).any(), "Abs amp")))# 最大值
    
    Epo_DroPA[fn] = epoch.copy().drop_bad(reject=reject).drop_bad(reject=dict(eeg=lambda x: ((np.abs(x) >75).any(), "Abs amp")))
    
A._data = A._data/10**6    
A = Epo_DroP['姜雷20240223刺激后_EEG.pkl']
A.plot_psd()
A.plot()
Epochs_use = {k:Epochs[k] for k in file_dt if len(Epochs[k])>30}    
    #A.copy().plot_drop_log()
    


