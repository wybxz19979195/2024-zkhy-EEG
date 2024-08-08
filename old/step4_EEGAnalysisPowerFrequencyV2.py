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
import pingouin as pg
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
Result= {}
Epochs_use = {k:Epoch_use[k] for k in Epoch_use.keys() if len(Epoch_use[k])>20}
f_names = [i for i in Epochs_use.keys()]
# 计算功率
Evokes = {fn: Epochs_use[fn].average() for fn in f_names}
Evo_power = {k:Evokes[k].compute_psd() for k in Evokes.keys()}
Savepkl(result_save,'step3_Evoke-PowerV2.pkl',Evo_power)
# 时频分析
freqs = np.arange(1, 100, 1)  # 1到50Hz
n_cycles = freqs / 2.
Epo_TF = {k:np.mean(mne.time_frequency.tfr_array_morlet(Epochs_use[k].get_data(),
        sfreq=250, freqs=freqs, n_cycles=n_cycles, output='power'),axis=0) for k in f_names}
Savepkl(result_save,'step4_Epochs-PowerTimeFrequencyMorlet.pkl',Epo_TF)
    p_dt = np.mean(np.stack([Pow_D[d][k] for k in Pow_D[d].keys() if s in k],axis=0),axis=0)
    data = np.log(np.mean(p_dt,axis=0)*10**12)
    pn = d+s+'Time-Frequency Analysis'
    plt.figure()
    plt.imshow(data, aspect='auto', origin='lower',extent=[0, 2,1, 100],vmax=8)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(pn)
    plt.colorbar(label='Power')
    plt.savefig(file_pic+pn+s+'-'+f+'.png') 
# %% 计算功率,相对功率，ROI等
Evo_power = Loadpkl(result_save,'step3_Evoke-PowerV2.pkl')
# fre-band
Fre = {'1Delta':[1,4],'2Theta':[4,8],'3Alpha':[8,12],
       '4Beta':[12,30],'5Gamma':[30,100],'6Whole':[1,100]}
F_name = [f for f in Fre.keys()]
# Roi
Ch_name = {k:Evo_power[k].copy().ch_names for k in Evo_power.keys()}
Roi_name = ['Frontal','Central','Parietal','Temporal','Occipital']
Roi_ch = {r:[i for i in Ch_name[f_names[16]] if i[0]==r[0]] for r in Roi_name}

Evo_PowFre = {k:{F:np.array([Evo_power[k]._data[:,i] for i in range(Evo_power[k]._data.shape[1]) 
            if Fre[F][0]<=Evo_power[k]._freqs[i]<Fre[F][1]])for F in Fre.keys()} for k in Evo_power.keys()}
Evo_K = [k for k in Evo_power.keys() if len(k)>3]
Evo_PF = {f:{k:np.sum(Evo_PowFre[k][f],axis=0) for k in Evo_K} for f in Fre.keys()}
Evo_PFM = {f:{k:np.mean(Evo_PF[f][k]) for k in Evo_K} for f in Fre.keys()}
Evo_PFRoi = {f:{r:[np.array([Evo_PF[f][k][ch] for ch in range(Evo_PF[f][k].shape[0]) 
                if Ch_name[k][ch] in Roi_ch[r]]).mean()
                for k in Evo_K] for r in Roi_name} for f in Fre.keys()}
# 相对频率
Evo_PFR = {f:{k:Evo_PF[f][k]/Evo_PF['6Whole'][k] for k in Evo_K}for f in F_name}
Evo_PFMR = {f:{k:Evo_PFM[f][k]/Evo_PFM['6Whole'][k] for k in Evo_K} for f in F_name}
Evo_PFRoiR = {f:{r:[Evo_PFRoi[f][r][i]/Evo_PFRoi['6Whole'][r][i] 
            for i in range(len(Evo_PFRoi[f][r]))] for r in Roi_name} for f in F_name}

# %% plot
# 1全脑平均
dt = Evo_PFM.copy()
pn = 'step4-FrePower-'

dt = Evo_PFMR.copy()
pn = 'step4-FrePowerRatio-'

D = pd.DataFrame(np.zeros((len(dt[F_name[0]])*len(Fre),4)),columns=['Power','Date','Stimulation','Fre-Band'])
D['Power'] = np.concatenate([[v for v in dt[f].values()] for f in Fre.keys()],axis=0)
D['Date'] = np.concatenate([[k[k.find('2024')+5: k.find('2024')+8] for k in dt[f].keys()] for f in Fre.keys()],axis=0)
D['Stimulation'] = np.concatenate([[k[k.find('刺激'): k.find('刺激')+3] for k in dt[f].keys()] for f in Fre.keys()],axis=0)
D['Fre-Band'] = np.concatenate([[f]*len(dt[f]) for f in Fre.keys()],axis=0)      
s = '刺激前'
for f in F_name : 
    data = D.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']==s]
    data = data[data['Fre-Band']==f]
    data= data[data['Date'].isin(['225', '308', '617'])]
    #data['Power'] = data['Power']*10**12
    plt.figure()
    sns.barplot(x='Date', y='Power', hue='Stimulation', data=data)
    for i, value in enumerate(data['Power']):
        plt.text(i, value, f'{value:.2f}', ha='center')
    plt.title(s+f)
    plt.savefig(file_pic+pn+s+'-'+f+'V3.png') 
# 2 channel 箱形图+散点图+方差分析
dt = Evo_PF.copy()
pn = 'step4-FrePowerChannel-'

dt = Evo_PFR.copy()
pn = 'step4-FreRatioChannel-'

D_Ch = pd.DataFrame(np.zeros((sum(len(v) for v in dt[F_name[0]].values())*len(F_name), (len(D.columns)))), columns=D.columns)       
D_Ch['Power'] = np.concatenate([np.concatenate([dt[f][k] for k in Evo_K]) for f in F_name])
D_Ch['Date'] = np.concatenate([np.hstack([[k[k.find('2024')+5: k.find('2024')+8]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()])
D_Ch['Stimulation'] = np.concatenate([np.hstack([[k[k.find('刺激'): k.find('刺激')+3]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()],axis=0)
D_Ch['Fre-Band'] = np.concatenate([[f]*int(D_Ch.shape[0]/len(F_name)) for f in Fre.keys()],axis=0) 
D_Ch['Channel'] = np.concatenate([np.hstack([Ch_name[k] for k in Evo_K])] * 6)
s = '刺激前'
Result_Avo = {}
for f in F_name : 
    data = D_Ch.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']==s]
    data = data[data['Fre-Band']==f]
    data= data[data['Date'].isin(['225', '308', '617'])]
    # 方差分析
    Result_Avo[f+'_Anona'] = pg.anova(data, dv='Power', between='Date', detailed=True)
    Result_Avo[f+'_HSD'] = pg.pairwise_tukey(data, dv='Power', between='Date')

    plt.figure()
    sns.set_style("whitegrid")
    sns.boxplot(data=data,x='Date', y='Power',fill=False)
    sns.stripplot(x='Date', y='Power', data=data, size=3)
    plt.title(s+f)
    plt.savefig(file_pic+pn+s+'-'+f+'V3.png') 
Result['Power-Ratio_Anova'] = Result_Avo.copy()
Savepkl(result_save,'step4-PowerAnova.pkl',Result)
#  topomap
K_use = [k for k in f_names if s in k and k[k.find('2024')+5: k.find('2024')+8] in ['225','308','617']]
pn = 'step4-FrePowerTopo-'
K_bands = {'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8),'Alpha (8-12 Hz)': 
           (8, 12), 'Beta (12-30 Hz)': (12, 30),'Gamma (30-45 Hz)': (30, 100)}
K_vlim = {'Delta (1-4 Hz)': (0.92, 54.30), 'Theta (4-8 Hz)': (0.18, 5.39),'Alpha (8-12 Hz)': 
          (0.07, 2.77), 'Beta (12-30 Hz)': (0.01, 1.32),'Gamma (30-45 Hz)': (0.01, 2.00)}

pn = 'step4-FrePowerRatioTopo-'
Evo_PFR_topo = {f[1:]:{k:np.tile(Evo_PFR[f][k], (251, 1)).T/10**12 for k in f_names} for f in Evo_PFR.keys()}
K_vlimR = {'Delta (1-4 Hz)': (0, 0.9), 'Theta (4-8 Hz)': (0,0.4),'Alpha (8-12 Hz)': 
          (0,0.4), 'Beta (12-30 Hz)': (0, 0.3),'Gamma (30-45 Hz)': (0, 0.7)}

for k in K_use:
    for i in K_vlim.keys():
        dt = Evo_power[k].copy()
        # Power
        fig = dt.copy().plot_topomap(bands = {i:K_bands[i]},vlim=K_vlim[i])
        fig.savefig(file_pic+pn+k+'-'+i+'.png') 
        #Power ration
        dt._data = np.tile(Evo_PFR_topo[i[0:i.find('(')-1]][k], (251, 1))
        fig = dt.plot_topomap(bands = {i:K_bands[i]},vlim=K_vlimR[i])
        fig.savefig(file_pic+pn+k+'-'+i+'.png') 
# %% plot ROI: 条形图 
dt = Evo_PFRoi.copy()
pn = 'step4-FrePowerRoi-'

dt = Evo_PFRoiR.copy()
pn = 'step4-FrePowerRoiRatio-'

D_Roi = pd.DataFrame(np.tile(D.values, (len(Roi_name), 1)), columns=D.columns)       
D_Roi['Power'] = np.concatenate([np.concatenate([dt[f][r] for f in F_name],axis=0) for r in Roi_name])
D_Roi['Roi'] = np.concatenate([[r]*len(f_names)*len(F_name) for r in Roi_name])
for f in F_name : 
    data = D_Roi.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']==s]
    data = data[data['Fre-Band']==f]
    data= data[data['Date'].isin(['225', '308', '617'])]
    plt.figure()
    sns.barplot(x='Date', y='Power', hue='Roi', data=data)
    plt.title(s+f)
    plt.savefig(file_pic+pn+s+'-'+f+'V3.png') 
    

