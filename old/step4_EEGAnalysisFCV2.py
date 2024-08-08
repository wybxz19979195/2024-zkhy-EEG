# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:32:17 2024

@author: ZKHY
"""
# 功能连接
# Coherence

import mne_connectivity,os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import mne
from mne_connectivity.viz import plot_sensors_connectivity,plot_connectivity_circle
import pingouin as pg
# 支持显示汉语
plt.rcParams['font.sans-serif'] = ['Fangsong']
plt.rcParams['axes.unicode_minus'] = False

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
Epochs_use = Loadpkl(result_save,'step3_EpochUsefulV2.pkl') 
# FC:coherence
Epo_FCC = {k:mne_connectivity.spectral_connectivity_epochs(
    Epochs_use[k],method='coh') for k in Epochs_use.keys()}
#imcoh: Imaginary part of Coherency (imcoh): 相干性的虚部
Epo_FCI = {k:mne_connectivity.spectral_connectivity_epochs(
    Epochs_use[k],method='imcoh') for k in Epochs_use.keys()}
# PLI:Phase Lag Index (PLI): 一种度量信号间因果关系的指标
Epo_FCP = {k:mne_connectivity.spectral_connectivity_epochs(
    Epochs_use[k],method='plv') for k in Epochs_use.keys()}
Savepkl(result_save,'step4_EpochsUse-FcCoherenceV2.pkl',Epo_FCC)
Savepkl(result_save,'step4_EpochsUse-FcCoherenceImaginary.pkl',Epo_FCI)
Savepkl(result_save,'step4_EpochsUse-FcCoherencePhaseLagIndex.pkl',Epo_FCP)
# %% load FC and calculate
Epo_FCC = Loadpkl(result_save,'step4_EpochsUse-FcCoherenceV2.pkl')
Epo_FCI = Loadpkl(result_save,'step4_EpochsUse-FcCoherenceImaginary.pkl')
Epo_FCP = Loadpkl(result_save,'step4_EpochsUse-FcCoherencePhaseLagIndex.pkl')
# Fre-band FC
Epo_Fc = Epo_FCI.copy() 
Result ={}
Fre = {'1Delta':[1,4],'2Theta':[4,8],'3Alpha':[8,12],
       '4Beta':[12,30],'5Gamma':[30,100],'6Whole':[1,100]}
F_name = [f for f in Fre.keys()]
Epo_Fc = Loadpkl(result_save,'step4_EpochsUse-FcCoherenceV2.pkl')
f_names = [k for k in Epo_Fc.keys()]
Ch_name = {k:Epochs_use[k].copy().ch_names for k in f_names}
FC_Fre = {k:{F:np.mean(np.array([Epo_Fc[k]._data[:,i] for i in range(Epo_Fc[k]._data.shape[1]) 
        if Fre[F][0]<=Epo_Fc[k].freqs[i]<Fre[F][1]]),axis=0).reshape(
        len(Ch_name[k]), len(Ch_name[k]))for F in Fre.keys()} for k in Epo_Fc.keys()}
FC_FreCh= {f:{k:FC_Fre[k][f][np.tril_indices(FC_Fre[k][f].shape[0],k=-1)] for k in f_names} for f in F_name}
FC_FreM = {f:{k:FC_FreCh[f][k].mean() for k in f_names} for f in F_name}              
# Roi fc
Roi_name = ['Frontal','Central','Parietal','Temporal','Occipital']
Roi_ch = {r:[i for i in Ch_name[f_names[16]] if i[0]==r[0]] for r in Roi_name}
Roi_pair =[(r,r) for r in Roi_name]+ [i for i in itertools.combinations(Roi_name, r=2)]
FC_FreRoi = {R1+'-'+R2:{F:{k:np.array([FC_Fre[k][F][ch1,ch2] 
            for ch1 in range(len(Ch_name[k]))for ch2 in range(len(Ch_name[k]))
            if Ch_name[k][ch1] in Roi_ch[R1] and Ch_name[k][ch2] in Roi_ch[R2]and ch1>ch2]).mean()
            for k in f_names} for F in F_name}  for R1 in Roi_name for R2 in Roi_name}  
# 平均相同roi名称，不同顺序的fc
FC_FreRoi2 = {i[0]+'-'+i[1]:{F:(pd.DataFrame([v for v in FC_FreRoi[i[0]+'-'+i[1]][F].values()])+
     pd.DataFrame([v for v in FC_FreRoi[i[1]+'-'+i[0]][F].values()]))/2 for F in F_name} for i in Roi_pair}
#Savepkl(result_save,'step4_EpochsUse-FcCoherenceRoiV2.pkl',FC_FreRoi2)

# %% 全脑平均条形图
dt = FC_FreM.copy()
pn = 'step4-FCCohMean-'
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
   data= data[data['Date'].isin(['223', '308', '617'])]
   #data['Power'] = data['Power']*10**12
   plt.figure()
   sns.barplot(x='Date', y='Power', hue='Stimulation', data=data)
   for i, value in enumerate(data['Power']):
       plt.text(i, value, f'{value:.2f}', ha='center')
   plt.title(s+f)
   plt.savefig(file_pic+pn+s+'-'+f+'V3.png') 
# %% Chaneel
dt = FC_FreCh.copy()
pn = 'step4-FreFCCohChannel-'
D_Ch = pd.DataFrame(np.zeros((sum(len(v) for v in dt[F_name[0]].values())*len(F_name), (len(D.columns)))), columns=D.columns)       
D_Ch['Power'] = np.concatenate([np.concatenate([dt[f][k] for k in f_names]) for f in F_name])
D_Ch['Date'] = np.concatenate([np.hstack([[k[k.find('2024')+5: k.find('2024')+8]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()])
D_Ch['Stimulation'] = np.concatenate([np.hstack([[k[k.find('刺激'): k.find('刺激')+3]]*dt[f][k].shape[0] for k in dt[f].keys()]) for f in Fre.keys()],axis=0)
D_Ch['Fre-Band'] = np.concatenate([[f]*int(D_Ch.shape[0]/len(F_name)) for f in Fre.keys()],axis=0) 
s = '刺激前'
Result_Avo = {}
for f in F_name : 
    data = D_Ch.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']==s]
    data = data[data['Fre-Band']==f]
    data= data[data['Date'].isin(['225', '308', '617'])]
    # 统计分析
    Result_Avo[f+'_Anona'] = pg.anova(data, dv='Power', between='Date', detailed=True)
    Result_Avo[f+'_HSD'] = pg.pairwise_tukey(data, dv='Power', between='Date')
    # 箱图+散点图
    plt.figure()
    sns.set_style("whitegrid")
    sns.boxplot(data=data,x='Date', y='Power',fill=False)
    sns.stripplot(x='Date', y='Power', data=data, size=3)
    plt.title(s+f)
    plt.savefig(file_pic+pn+s+'-'+f+'V2.png') 
# channel fc distrabution
K_use = [k for k in f_names if s in k and k[k.find('2024')+5: k.find('2024')+8] in ['223','225','617']]
dt = FC_FreCh.copy()
f = F_name[4]
for f in F_name:
    for k in K_use:
        plt.figure()
        plt.hist(dt[f][k])
        plt.title(k+f+'Data Distribution')
# 弦图 每个通道
pn = 'step4-FCCohCircleChannel-'
dt = FC_Fre.copy()
for k in K_use:
    for f in F_name:
        plot_connectivity_circle(dt[k][f],node_names=Ch_name[k],
        vmin=0,vmax=1,facecolor='black',title = f+k[0:-4])
        plt.savefig(file_pic+pn+s+'-'+f+'.svg', facecolor='black') 
        
# %% Roi 热图
dt = FC_FreRoi2.copy()
FC_FreMRoiM = {F:{} for F in Fre.keys()} 
D = pd.DataFrame(np.zeros((len(Roi_name),len(Roi_name))),columns=Roi_name)
D.index = Roi_name
for F in Fre.keys():
    print(F)
    for i in range(len(f_names)):
        for R in dt.keys():
            r1 = Roi_name.index(R[0:R.find('-')])
            r2 = Roi_name.index(R[R.find('-')+1:])
            D.iloc[r1,r2] = dt[Roi_name[r1]+'-'+Roi_name[r2]][F].iloc[i,0]
            D.iloc[r2,r1] = dt[Roi_name[r1]+'-'+Roi_name[r2]][F].iloc[i,0]
        FC_FreMRoiM[F][f_names[i]] = D.copy()

K_use = [k for k in f_names if s in k and k[k.find('2024')+5: k.find('2024')+8] in ['225','308','617']]
dt = FC_FreMRoiM
for F in Fre.keys():
    for k in range(len(K_use)):
        data = dt[F][K_use[k]]
        pn = 'step4_HMFCCoherenceRoi-'+F+'-'+K_use[k]
        plt.figure()
        sns.heatmap(data,vmax=1,vmin=0, cmap="OrRd")
        plt.xticks(rotation=45)
        plt.yticks( rotation=45)
        plt.title(F+'-'+K_use[k])
        plt.savefig(file_pic+pn+'.png')

