# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:12:37 2024

@author: ZKHY
"""
# ICA and Epoch
import mne,os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, lfilter
import scipy.signal as signal
from mne.preprocessing import ICA
import ast
from pypinyin import lazy_pinyin as py
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
wd = r'D:\zkhy\jl_hmcx_zsqy\\'
os.chdir(wd)
result_save = '3result\\'
reseult_pt = '3result\\plot\\'

Raw_Fil = Loadpkl(result_save, 'step1_EEGFilter.pkl')
file_dt = [k for k in Raw_Fil.keys()]
dt_ana = pd.read_excel(result_save+'分析记录.xlsx')
# %% 提取epoch
sfreq = 250
duration = 2
tmin=0
tmax=duration
Epochs = {}
for i in range(len(file_dt)):
    fn = file_dt[i]
    
    if '姜雷' not in fn:
        print(fn)
        raw =Raw_Fil[fn]
        t_total = raw.times[-1]
        
                 
        raw_rest = raw.copy()
        n_samples = len(raw_rest)  # 总采样点
        n_seconds = n_samples / sfreq   # 总秒数
        onset = np.arange(0, n_seconds, duration)  # 生成标记时间点（秒）
        description = 'R'
        # 创建注释对象Built-in muta
        annotations = mne.Annotations(onset=onset, duration=duration, description=description)
        # 将注释添加到 Raw 对象中
        Raw_mark = raw_rest.copy().set_annotations(annotations)
        events,event_dict = mne.events_from_annotations(Raw_mark)
        Epochs[fn] = mne.Epochs(Raw_mark, events=events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True,
                                baseline=(0, 0))
# %% 删除坏段
file_dt = [k for k in Epochs.keys()]
Epo_DroPA = {}
reject = dict(eeg=40e-6) # 删除坏段的标准 PTP最大峰值差
reject_PF = 75e-6 # 删除坏段的标准 PF峰值频率 
for fn in Epochs.keys():
    Epo_DroPA[fn] = Epochs[fn].copy().drop_bad(reject=reject)
    # 删除绝对值超过PF的段
    abs_reject_indices = [i for i, e in enumerate(Epo_DroPA[fn].get_data()) if (np.abs(e) > reject_PF).any()]
    Epo_DroPA[fn] = Epo_DroPA[fn].drop(abs_reject_indices, reason='ABS_AMP')
Epo_bad = {fn:Epochs[fn] for fn in Epochs.keys() if len(Epo_DroPA[fn])<20}
for fn in Epo_bad.keys():
    #Epochs['王后兰-刺激前-20240606'].plot_psd()_drop_log()
    raw = Epochs[fn].copy()
    bad_ch = [dt_ana['坏导'][i]  for i in range(dt_ana.shape[0])if dt_ana['记录次数'][i]==fn][0]
    raw.info['bads'] = ast.literal_eval(bad_ch)
    raw = raw.copy().drop_bad(reject=reject)
    abs_reject_indices = [i for i, e in enumerate(raw.get_data()) if (np.abs(e) > reject_PF).any()]
    Epo_DroPA[fn] = raw.drop(abs_reject_indices, reason='ABS_AMP')
    
    raw.plot_psd_topomap()
    plt.title(fn)
Savepkl(result_save,'//step2_EpochDropBad-2s.pkl', Epo_DroPA)   
# ICA and 坏导处理
Epo_ICA = {}
for n in Epo_DroPA.keys():
    Epo_DroPA[n].plot_psd_topomap()
    plt.title(n)
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    Epo_ICA[n] = ica.fit(Epo_DroPA[n])
    fig = Epo_ICA[n].plot_components(title = n)
    fig.savefig(result_save+'//plot//' +'step1_EpoRestICA-'+n+'V2.png', dpi=300)
    plt.close(fig)
Savepkl(result_save,'//step2_EpochRestICA.pkl', Epo_ICA) 
Raw_Fil['王后兰-刺激前-20240606'].plot()
# 手动识别ICA和坏导
Epo_DroPA = Loadpkl(result_save,'//step2_EpochDropBad-2s.pkl')
Epo_ICA = Loadpkl(result_save,'//step2_EpochRestICA.pkl')
file_names = [k for k in Epo_DroPA.keys()]    

n = file_names[8]
print(n)
raw = Epo_DroPA[n].copy()
raw.plot_psd()
raw.plot()
raw.plot_psd_topomap()
ica = Epo_ICA[n].copy()
d = [0,1,2,3,6]#,17,19
d = []
ica.plot_properties(raw, picks=d)  # 选择剔除成分进行可视化
ica.exclude = d

dt = ica.apply(raw.copy())
dt.plot_psd()
dt.plot()
dt.plot_psd_topomap()
