# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:34:44 2024

@author: ZKHY
"""
# 20240223 和 20240225的数据都需要将原始数据乘上223.5174
import mne,os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
# 支持显示汉语
plt.rcParams['font.sans-serif'] = ['Fangsong']
plt.rcParams['axes.unicode_minus'] = False
#matplotlib.use('TkAgg')
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
result_save = '3result\\'
file_pic = '3result\\plot\\'
file_dt = os.listdir(result_save)
Epoch_use = Loadpkl(result_save,'step3_EpochUsefulV2.pkl')  
# 筛选包含“刺激前”字符的Evoked对象
Epochs_use = {k:Epoch_use[k] for k in Epoch_use.keys() if len(Epoch_use[k])>20}
f_names = [i for i in Epochs_use.keys()]
Evokes = {fn: Epochs_use[fn].average() for fn in f_names}
Evo_power = {k:Evokes[k].compute_psd() for k in Evokes.keys()}
Savepkl(result_save,'step3_Evoke-PowerV2.pkl',Evo_power)
# 绘制条形图：power随时间变化
Evo_power = Loadpkl(result_save,'step3_Evoke-Power.pkl')
fn = '姜雷-20240617-刺激前_EEG.edf'
Evo_power[fn] = Evokes[fn].compute_psd()
del Evo_power['姜雷-20240415-刺激前-任务态-5分钟_EEG.edf']
del Evo_power['姜雷-20240415-刺激后-任务态-5分钟_EEG.edf']
del Evo_power['姜雷20240223刺激前_EEG.edf']
del Evo_power['姜雷-20240409-刺激前-10分钟_EEG.edf']
Pow_Bef = np.mean(np.array([Evo_power[k]._data for k in Evo_power.keys() if '刺激前' in k]),axis=0)
Pow_Aft = np.mean(np.array([Evo_power[k]._data for k in Evo_power.keys() if '刺激后' in k]),axis=0)        

# Roi
Ch_name = Evo_power[f_names[0]].copy().ch_names
Roi_name = ['Frontal','Central','Parietal','Temporal','Occipital']
Roi_ch = {r:[i for i in Ch_name if i[0]==r[0]] for r in Roi_name}
Fre = {'1Delta':[1,4],'2Theta':[4,8],'3Alpha':[8,12],
       '4Beta':[12,30],'5Gamma':[30,100],'6Whole':[1,100]}
F_name = [f for f in Fre.keys()]
Evo_PowFre = {k:{F:np.array([Evo_power[k]._data[:,i] for i in range(Evo_power[k]._data.shape[1]) if Fre[F][0]<=Evo_power[k]._freqs[i]<Fre[F][1]])
                 for F in Fre.keys()} for k in Evo_power.keys()}
Evo_K = [k for k in Evo_power.keys() if len(k)>3]
Evo_PowFreM = {f:{} for f in Fre.keys()}
Evo_PowFreMRoi = {f:{} for f in Fre.keys()}
for F in Fre.keys():
    for k in Evo_K:
        Evo_PowFreM[F][k] = Evo_PowFre[k][F].sum()
        Evo_PowFreMRoi[F] = {r:[np.array([Evo_PowFre[k][F][:,ch] for ch in range(Evo_PowFre[k][F].shape[1]) if Ch_name[ch] in Roi_ch[r]]).sum()for k in Evo_K] for r in Roi_name}

# 相对频率
Evo_PowFreMR = {f:{k:Evo_PowFreM[f][k]/Evo_PowFreM['6Whole'][k] for k in Evo_K} for f in F_name}
Evo_PowFreMRoiR = {f:{r:[Evo_PowFreMRoi[f][r][i]/Evo_PowFreMRoi['6Whole'][r][i] 
                for i in range(len(Evo_PowFreMRoi[f][r]))] for r in Roi_name} for f in F_name}

# plot
# 全脑平均
dt = Evo_PowFreM.copy()
pn = 'step4-FrePower-'
#data['Power'] = data['Power']*10**9
dt = Evo_PowFreMR.copy()
pn = 'step4-FrePowerRatio-'
#data = data[data['Fre-Band']!='6Whole']
D = pd.DataFrame(np.zeros((len(dt[F_name[0]])*len(Fre),4)),columns=['Power','Date','Stimulation','Fre-Band'])
D['Power'] = np.concatenate([[v for v in dt[f].values()] for f in Fre.keys()],axis=0)
D['Date'] = np.concatenate([[k[k.find('2024')+5: k.find('2024')+8] for k in dt[f].keys()] for f in Fre.keys()],axis=0)
D['Stimulation'] = np.concatenate([[k[k.find('刺激'): k.find('刺激')+3] for k in dt[f].keys()] for f in Fre.keys()],axis=0)
D['Fre-Band'] = np.concatenate([[f]*len(dt[f]) for f in Fre.keys()],axis=0)      
s = '刺激后'
for f in F_name : 
    data = D.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']==s]
    data = data[data['Fre-Band']==f]
    data['Power'] = data['Power']*10**9
    plt.figure()
    sns.barplot(x='Date', y='Power', hue='Fre-Band', data=data)
    for i, value in enumerate(data['Power']):
        plt.text(i, value, f'{value:.2f}', ha='center')
    plt.title(s+f)
    plt.savefig(file_pic+pn+s+'-'+f+'V2.png') 
#
# plot ROI    
dt = Evo_PowFreMRoi.copy()
pn = 'step4-FrePowerRoi-'

dt = Evo_PowFreMRoiR.copy()
pn = 'step4-FrePowerRoiRatio-'

D_Roi = pd.DataFrame(np.tile(D.values, (len(Roi_name), 1)), columns=D.columns)       
D_Roi['Power'] = np.concatenate([np.concatenate([dt[f][r] for f in F_name],axis=0) for r in Roi_name])
D_Roi['Roi'] = np.concatenate([[r]*len(f_names)*len(F_name) for r in Roi_name])
for f in F_name : 
    data = D_Roi.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']==s]
    data = data[data['Fre-Band']==f]
    #data = data[data['Roi']!='Central']
    data['Power'] = data['Power']*10**9
    plt.figure()
    sns.barplot(x='Date', y='Power', hue='Roi', data=data)
    plt.title(s+f)
    plt.savefig(file_pic+pn+s+'-'+f+'V2.png') 