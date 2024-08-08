# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:34:44 2024

@author: ZKHY
"""
import mne,os,ast
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
wd = r'D:\zkhy\jl_hmcx_zsqy\\'
os.chdir(wd)
result_save = '3result\\'
file_pic = '3result\\plot\\'
Epochs = Loadpkl(result_save,'step3_EpochsUseRest.pkl')
f_names = [k for k in Epochs.keys()]
# Evoke
Evokes = {k:Epochs[k].average() for k in f_names}   
# 功率
Evo_power = {k:Evokes[k].compute_psd() for k in f_names}
# fre-band
Fre = {'1Delta':[1,4],'2Theta':[4,8],'3Alpha':[8,12],
       '4Beta':[12,30],'5Gamma':[30,100],'6Whole':[1,100]}
F_name = [f for f in Fre.keys()]
# Roi
Ch_name = {k:Evo_power[k].copy().ch_names for k in f_names}
Roi_name = ['Frontal','Central','Parietal','Temporal','Occipital']
Roi_ch = {r:[i for i in Ch_name[f_names[0]] if i[0]==r[0]] for r in Roi_name}

Evo_PowFre = {k:{F:np.array([Evo_power[k]._data[:,i] for i in range(Evo_power[k]._data.shape[1]) 
            if Fre[F][0]<=Evo_power[k]._freqs[i]<Fre[F][1]])for F in Fre.keys()} for k in Evo_power.keys()}

Evo_PF = {f:{k:np.sum(Evo_PowFre[k][f],axis=0) for k in f_names} for f in Fre.keys()}
Evo_PFM = {f:{k:np.mean(Evo_PF[f][k]) for k in f_names} for f in Fre.keys()}
Evo_PFRoi = {f:{r:{k:np.array([Evo_PF[f][k][ch] for ch in range(Evo_PF[f][k].shape[0]) 
                if Ch_name[k][ch] in Roi_ch[r]]).mean()
                for k in f_names} for r in Roi_name} for f in Fre.keys()}
# 相对频率
Evo_PFR = {f:{k:Evo_PF[f][k]/Evo_PF['6Whole'][k] for k in f_names}for f in F_name}
Evo_PFMR = {f:{k:Evo_PFM[f][k]/Evo_PFM['6Whole'][k] for k in f_names} for f in F_name}
Evo_PFRoiR = {f:{r:{k:Evo_PFRoi[f][r][k]/Evo_PFRoi['6Whole'][r][k] 
            for k in f_names} for r in Roi_name} for f in F_name}

# %% plot
# 1全脑平均

dt = Evo_PFM.copy()
pn = 'step4-FP2-'

dt = Evo_PFMR.copy()
pn = 'step4-FPR2-'

columns=['name','Power','Stimulation','Fre-Band','Date']
D = pd.DataFrame(np.zeros((len(dt[F_name[0]])*len(Fre),len(columns))),columns=columns)
D['name'] = np.concatenate([[k[0:3] for k in dt[f].keys()] for f in Fre.keys()])
D['Power'] = np.concatenate([[v for v in dt[f].values()] for f in Fre.keys()])
D['Stimulation'] = np.concatenate([[k[4:7] for k in dt[f].keys()] for f in Fre.keys()])
D['Fre-Band'] = np.concatenate([[f]*len(dt[f]) for f in Fre.keys()],axis=0)      
D['Date'] = np.concatenate([[k[13:] for k in dt[f].keys()] for f in Fre.keys()])
Result_Avo = {}
x = 'Date'
hue = 'Stimulation'
col = 'name'
for f in F_name : 
    data = D.copy()
    data = data.sort_values(by=[x, hue])
    data = data[data['Fre-Band']==f]
    sns.catplot(kind="bar",col=col,x=x, y='Power', 
                hue=hue, data=data)
    plt.title(f)
    plt.savefig(file_pic+pn+'-'+f+'.png') 
    
    
# %%2 channel 箱形图+散点图+方差分析
dt = Evo_PF.copy()
pn = 'step4-FPChannel-'

dt = Evo_PFR.copy()
pn = 'step4-FPRChannel-'

D_Ch = pd.DataFrame(np.zeros((sum(len(v) for v in dt[F_name[0]].values())*len(F_name), (len(D.columns)))), columns=D.columns)       
D_Ch['Power'] = np.concatenate([np.concatenate([dt[f][k] for k in f_names]) for f in F_name])
D_Ch['name'] = np.concatenate([np.hstack([[k[0:3]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()])
D_Ch['Stimulation'] = np.concatenate([np.hstack([[k[4:7]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()],axis=0)
D_Ch['Date'] = np.concatenate([np.hstack([[k[13:]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()],axis=0)
D_Ch['Fre-Band'] = np.concatenate([[f]*int(D_Ch.shape[0]/len(F_name)) for f in Fre.keys()],axis=0) 
D_Ch['Channel'] = np.concatenate([np.hstack([Ch_name[k] for k in f_names])] * 6)
x = 'Date'
hue = 'Stimulation'
col = 'name'
Result_Avo = {}
for f in F_name : 
    data = D_Ch.copy()
    data = data.sort_values(by=[x,hue])
    data = data[data['Fre-Band']==f]
    # 方差分析
   
    plt.figure()
    sns.set_style("whitegrid")
    sns.boxplot(data=data,x='Date', y='Power',fill=False)
    sns.stripplot(x='Date', y='Power', data=data, size=3)
    plt.title(f)
    plt.savefig(file_pic+pn+'-'+f+'V3.png') 
    
Result['Power-Ratio_Anova'] = Result_Avo.copy()
Savepkl(result_save,'step4-PowerAnova.pkl',Result)
# %% topomap

K_bands = {'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8),'Alpha (8-12 Hz)': 
           (8, 12), 'Beta (12-30 Hz)': (12, 30),'Gamma (30-45 Hz)': (30, 100)}
K_vlim = {'Delta (1-4 Hz)': (0.92, 54.30), 'Theta (4-8 Hz)': (0.18, 5.39),'Alpha (8-12 Hz)': 
          (0.07, 2.77), 'Beta (12-30 Hz)': (0.01, 1.32),'Gamma (30-45 Hz)': (0.01, 2.00)}

Evo_PFR_topo = {f[1:]:{k:np.tile(Evo_PFR[f][k], (251, 1)).T/10**12 for k in f_names} for f in Evo_PFR.keys()}
K_vlimR = {'Delta (1-4 Hz)': (0, 0.9), 'Theta (4-8 Hz)': (0,0.4),'Alpha (8-12 Hz)': 
          (0,0.4), 'Beta (12-30 Hz)': (0, 0.3),'Gamma (30-45 Hz)': (0, 0.7)}

pn = 'step4-FPTopo-'   

dt_topoP = {str(g)+'-'+str(d):np.mean(np.array([Evo_power.copy()[k]._data for k in Group.keys() 
    if int(k[0])==d and Group[k]==g]),axis=0) for g in [1,2] for d in [1,2,3]}

dt_topoPR = {f:{str(g)+'-'+str(d):np.array([Evo_PFMR[f][k] for k in Group.keys()if int(k[0])==d 
     and Group[k]==g]).mean() for g in [1,2] for d in [1,2,3]}for f in F_name}

for k in dt_topo.keys():
    data = Evo_power['1-01006ADTACS-06.pkl'].copy()
    data._data = dt_topoP[k]
    
    fig = data.plot_topomap()
    plt.title(k)
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
pn = 'step4-FP-Roi-'

dt = Evo_PFRoiR.copy()
pn = 'step3-FPR-Roi-'

D_Roi = pd.DataFrame(np.tile(D.values, (len(Roi_name), 1)), columns=D.columns)       
D_Roi['Power'] = np.concatenate([np.concatenate([[dt[f][r][k] for k in f_names] for f in F_name]) for r in Roi_name])
D_Roi['Roi'] = np.concatenate([[r]*len(f_names)*len(F_name) for r in Roi_name])

for f in F_name : 
    data = D_Roi.copy()
    data = data.sort_values(by=[x, 'Fre-Band'])
    data = data[data['Fre-Band']==f]

    sns.catplot(kind="bar",col='Roi',x=x, y='Power', 
                hue='Stimulation', data=data)
    plt.title(f)
    plt.savefig(file_pic+pn+'-'+f+'.png') 
    

