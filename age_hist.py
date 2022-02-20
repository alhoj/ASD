#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:52:47 2021

@author: ja151
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'cluster': '/autofs/cluster/transcend/jussi/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

f = open('%s/p/subjects.p' % paths['out'], 'rb')
sub_info = pickle.load(f)
f.close()

exclude = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for sub in exclude:
    ind = sub_info['sub_ID'].index(sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
     if len(sub_info[key])==n_subs]
    
n_subs = len(sub_info['sub_ID'])
TD_ages = [sub_info['age'][i] for i in range(n_subs) if sub_info['ASD'][i]=='No']
ASD_ages = [sub_info['age'][i] for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
n_TD = len(TD_ages)
n_ASD = len(ASD_ages)

bin_edges = np.arange(7,19,2)
# TD_ages = [age if age>8 else age+1 for age in TD_ages]
# ASD_ages = [age if age>8 else age+1 for age in ASD_ages]
# bin_edges = [8, 10, 12, 14, 16, 18] # lower freqs

plt.rcParams['font.family'] = 'sans serif'
plt.hist([TD_ages, ASD_ages], bin_edges, color=['lightgreen', 'orchid'])
plt.xticks(bin_edges)
plt.legend(['TD', 'ASD'], prop={'weight': 'bold'})
plt.xlabel('Age', fontweight='bold')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('%s/figures/age_hist_%dTD_%dASD.png' % (paths['cluster'], n_TD, n_ASD), dpi=300)
plt.close() 