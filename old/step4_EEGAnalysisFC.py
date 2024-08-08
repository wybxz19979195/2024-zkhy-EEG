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
Fre = {'1Delta':[1,4],'2Theta':[4,8],'3Alpha':[8,12],
       '4Beta':[12,30],'5Gamma':[30,100],'6Whole':[1,100]}
F_name = [f for f in Fre.keys()]
Epo_Fc = {k:mne_connectivity.spectral_connectivity_epochs(
    Epochs_use[k],method='coh') for k in Epochs_use.keys()}
Savepkl(result_save,'step4_EpochsUse-FcCoherenceV2.pkl',Epo_Fc)

# 计算全Roi fc
Epo_Fc = Loadpkl(result_save,'step4_EpochsUse-FcCoherenceV2.pkl')
f_names = [k for k in Epo_Fc.keys()]
Ch_name = {k:Epochs_use[k].copy().ch_names for k in f_names}
Roi_name = ['Frontal','Central','Parietal','Temporal','Occipital']
Roi_ch = {r:[i for i in Ch_name if i[0]==r[0]] for r in Roi_name}
Roi_pair =[(r,r) for r in Roi_name]+ [i for i in itertools.combinations(Roi_name, r=2)]

Fre_band = {F:[f for f in Epo_Fc[f_names[0]].freqs if Fre[F][0]<=f<Fre[F][1]] for F in Fre.keys()}
FC_Fre = {k:{F:np.mean(np.array([Epo_Fc[k]._data[:,i] for i in range(Epo_Fc[k]._data.shape[1]) 
            if Fre[F][0]<=Epo_Fc[k].freqs[i]<Fre[F][1]]),axis=0).reshape(32, 32)
                 for F in Fre.keys()} for k in Epo_Fc.keys()}
FC_FreMRoi = {R1+'-'+R2:{} for R1 in Roi_name for R2 in Roi_name}

for R1 in Roi_name :
    for R2 in Roi_name:
        FC_FreMRoi[R1+'-'+R2] = {F:{k:np.array([FC_Fre[k][F][ch1,ch2] for ch1 in range(32) 
        for ch2 in range(32)  if Ch_name[ch1] in Roi_ch[R1] and Ch_name[ch2] in Roi_ch[R2] 
        and ch1>ch2]).mean()for k in f_names} for F in F_name}       
# 平均相同roi名称，不同顺序的fc
FC_FreMRoi2 = {i[0]+'-'+i[1]:{F:(pd.DataFrame([v for v in FC_FreMRoi[i[0]+'-'+i[1]][F].values()])+
     pd.DataFrame([v for v in FC_FreMRoi[i[1]+'-'+i[0]][F].values()]))/2 for F in F_name} for i in Roi_pair}
Savepkl(result_save,'step4_EpochsUse-FcCoherenceRoi.pkl',FC_FreMRoi2)


# %%画图
# 条形图
Evo_Kuse = [k for k in Epochs_use.keys()]
dt = FC_FreMRoi2.copy()
D = pd.DataFrame(np.zeros((len(dt[Roi_pair[0]][F_name[0]])*len(Fre),4)),columns=['Power','Date','Stimulation','Fre-Band'])
D['Power'] = np.concatenate([[v for v in dt[f].values()] for f in Fre.keys()],axis=0)
D['Date'] = np.concatenate([[k[k.find('2024')+5: k.find('2024')+8] for k in dt[f].keys()] for f in Fre.keys()],axis=0)
D['Stimulation'] = np.concatenate([[k[k.find('刺激'): k.find('刺激')+3] for k in dt[f].keys()] for f in Fre.keys()],axis=0)
D['Fre-Band'] = np.concatenate([[f]*len(dt[f]) for f in Fre.keys()],axis=0)      
for f in F_name : 
    data = D.copy()
    data = data.sort_values(by=['Date', 'Fre-Band'])
    data = data[data['Stimulation']=='刺激前']
    data = data[data['Fre-Band']==f]
    plt.figure()
    sns.barplot(x='Date', y='Power', hue='Fre-Band', data=data)
    #plt.ylim(0/10**10,5/10**12)
    plt.title('刺激前-'+f)   
# Now, visualize the connectivity in 3D:
k = f_names[0]
plot_sensors_connectivity(Epochs_use[k].info,Epo_Fc[k]._data[:,0].reshape(len(Ch_name[k]), len(Ch_name[k])))
plot_connectivity_circle(Epo_Fc[k]._data[:,0].reshape(len(Ch_name[k]), len(Ch_name[k])),node_names=Ch_name[k])
# 热图
dt_FreMRoiHM = {F:{} for F in Fre.keys()} 
D = pd.DataFrame(np.zeros((len(Roi_name),len(Roi_name))),columns=Roi_name)
D.index = Roi_name
for F in Fre.keys():
    print(F)
    for i in range(len(F_name)):
        k = F_name[i]
        for R in dt_FreMRoi2.keys():
            r1 =  Roi_name.index(R[0:R.find('-')])
            r2 = Roi_name.index(R[R.find('-')+1:])
            D.iloc[r1,r2] = dt_FreMRoi2[Roi_name[r1]+'-'+Roi_name[r2]][F][i]
            D.iloc[r2,r1] = dt_FreMRoi2[Roi_name[r1]+'-'+Roi_name[r2]][F][i]
        dt_FreMRoiHM[F][F_name[i]] = D.copy()
                    
for F in Fre.keys():
    for k in F_name:
        dt = dt_FreMRoiHM[F][k]
        pn = 'step4_HMFCCoherenceRoi-'+F+'-'+k
        plt.figure()
        sns.heatmap(dt,vmax=1,vmin=0, cmap="YlGnBu")
        plt.xticks(rotation=45)
        plt.yticks( rotation=45)
        plt.title(pn)
        plt.savefig(file_pic+pn+'.png')