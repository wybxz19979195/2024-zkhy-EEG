# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:49:07 2024

@author: ZKHY
"""

import seaborn as sns
import matplotlib.pyplot as plt
import mne,os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
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
wd = r'D:\zkhy\CD_TI\\'
os.chdir(wd)
result_save = '3result\\'
file_pic = '3result\\plot\\'
p_bh = {'WYLA':'001','LGYI':'002','LJHU':'004','PXRU':'005',
        'HSGE':'006','ZDHO':'007','KJZH':'008'}
p_name = [k for k in p_bh.keys()]
ID_del = ['KJZH_01A_TASK.pkl','HSGE_01A_TASK.pkl']
dt = Loadpkl(result_save,'step3_EvokeRestPowerV5'+'.pkl')
dt = {k:dt[k] for k in dt.keys() if k not in ID_del}
f_names = [k for k in dt.keys()]
Ch_names = dt[f_names[0]].ch_names
# 刺激后-刺激前
dt_A = {k:dt[k]._data for k in f_names if k[7]=='A' }
dt_B = {k:dt[k]._data for k in f_names if k[7]=='B' }
key_A = [k for k in dt_A.keys()]
key_B = [k for k in dt_B.keys()]
dt_BA = {key_B[i]:abs(dt_B[key_B[i]])-abs(dt_A[key_A[i]]) for i in range(len(key_A))}

# 平均被试
for n in p_name:
    data = dt[f_names[0]].copy()
    data._data = np.mean(np.array([dt_BA[k] for k in dt_BA.keys() if k[0:4]==n]),axis=0)
    data.plot_topomap()
    plt.savefig(file_pic+p_bh[n]+'step4-PF-刺激后-刺激前'+'.png') 

# 前/后10次刺激
dt1 = dt[f_names[0]].copy()
dt2 = dt[f_names[2]].copy()

dt1._data = np.mean(np.array([v for v in dt_A.values()]),axis=0)
dt2._data = np.mean(np.array([v for v in dt_B.values()]),axis=0)
# 全脑频谱图分布
fig, ax = plt.subplots(figsize=(10, 5))
dt1.plot_topo(axes=ax, color='black',fig_facecolor='white',axis_facecolor='white')
dt2.plot_topo(axes=ax, color='red',fig_facecolor='white',axis_facecolor='white')

# 全脑平均频谱图分布
fig, ax = plt.subplots(figsize=(10, 5))
dt1.copy().plot(axes=ax, average=True,spatial_colors=True, show=False, color='black')
dt2.copy().plot(axes=ax, average=True,spatial_colors=True, show=False, color='red')
ax.set_xlim(1, 45)
ax.set_ylim(-10, 15)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()

for ch in Ch_names:
    fig, ax = plt.subplots(figsize=(10, 5))
    line1 = dt1.copy().pick_channels([ch]).plot(axes=ax, average=True, 
                    spatial_colors=True, show=False, color='black')
    line2 = dt2.copy().pick_channels([ch]).plot(axes=ax, average=True, 
                        spatial_colors=True, show=False, color='red')
    for line in ax.get_lines():
        line.set_linewidth(1)
    ax.set_xlim(1, 45)
    ax.set_ylim(-10, 10)
    plt.title(ch)
    ax.tick_params(axis='both', which='major', labelsize=16)
# %%bar+散点
data = dt_step
plt.figure(figsize=(10, 6))
sns.stripplot(x=col,y='Power',hue=x,data=data,dodge=True,palette="Set2")
sns.barplot(x=col,y='Power',hue=x,data=data,capsize=.4,errorbar=("pi", 50),
            err_kws={"color": ".5", "linewidth": 2.5},linewidth=2.5, edgecolor=".5", facecolor=(0, 0, 0, 0))
plt.legend([], [], frameon=False)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

# 绘制 pointplot
plt.figure(figsize=(10, 6))
point = sns.pointplot(x=x,y='Power',hue=col,data=dt_step,
    dodge=True,ci='sd', palette="Set2")


# %% topomap
def Plot_topomap(data,t_ch,file_pic,fn):
    # 绘制差异分析的显著性的的地形图
    # data需要绘图的值
    # t_ch：字典每个通道的显著性,file_pic：图片保存的通道,图片名称
    fig = data.plot_topomap(bands = {'Delta (1-4 Hz)': (1, 4)},vlim=(-3, 3),show_names=True) 
    fig.set_size_inches(12, 10)
    for ax in fig.axes:
        for text in ax.texts:  # 遍历所有通道名称文本
            channel_name = text.get_text()
            # 添加单独的 * 标记
            if channel_name in [k for k in t_ch.keys()]:
                p_value = t_ch[channel_name]  # 获取对应的 p 值
                # 根据 p 值确定标记符号
                if 0.01 < p_value <= 0.055:
                    star = '*'
                elif 0.001 < p_value <= 0.01:
                    star = '**'
                elif p_value <= 0.001:
                    star = '***'
                else:
                    continue  # 跳过无显著性的通道
                ax.text(text.get_position()[0] + 0.007,text.get_position()[1] - 0.003,
                    star,fontsize=20,fontweight='bold',       
                    fontname='Arial', color='red', ha='center', va='center')
            text.set_fontsize(16)         
            text.set_fontweight('bold')  
            text.set_fontname('Arial')   
            text.set_y(text.get_position()[1] - 0.003)  
        cbar_ax = fig.get_axes()[-1]  # 获取 colorbar 的 axes 对象
        cbar_ax.tick_params(labelsize=22)  # 修改 colorbar 刻度字体大小
        cbar_ax.set_ylabel('') 
        plt.title('')
        #plt.show()
        plt.savefig(file_pic+fn+'.png')
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

