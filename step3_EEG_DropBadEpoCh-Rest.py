# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:37:11 2024

@author: ZKHY
"""
import mne,os
import pickle as pkl
import pandas as pd
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import ast
from scipy.signal import filtfilt, butter, lfilter
import scipy.signal as signal
from mne.time_frequency import tfr_multitaper
from mne.preprocessing import ICA
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
dt_ana = pd.read_excel(result_save+'分析记录.xlsx')
Epochs1 = Loadpkl(result_save, 'step2_EpochDropBad-2s.pkl')
Epo_ICA = Loadpkl(result_save, 'step2_EpochRestICA.pkl')
f_epo = [k for k in Epochs1.keys()]
Name = [i for i in dt_ana['记录次数']]
# mark
Epochs = {}
for n in f_epo:
    i = Name.index(n)
    print(n+'----'+dt_ana['记录次数'][i])
    if dt_ana['ICA'][i]==dt_ana['ICA'][i]:
        print(n+'----'+dt_ana['记录次数'][i])
        epo = Epochs1[n].copy()
        ica = Epo_ICA[n].copy()
        ica.exclude = ast.literal_eval(dt_ana['ICA'][i])
        Epochs[n] = ica.apply(epo.copy())
        epochs = Epochs[n]
        bad_ch = dt_ana['坏导'][i]
        if bad_ch==bad_ch: # 判断是否为空
            print(n+'--'+bad_ch)
            epochs.info['bads'] = ast.literal_eval(bad_ch) 
            Epochs[n] =epochs.interpolate_bads() # 插值坏导
            Epochs[n].plot_psd_topomap()
            plt.title(n)
            # epochs.drop_channels(epochs.info['bads']) # 删除坏导
Savepkl(result_save, 'step3_EpochsUseRest'+'.pkl', Epochs)
