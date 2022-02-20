#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:51:10 2020

@author: ja151
"""

import pickle
import numpy as np
from scipy import stats

sub_info = pickle.load(open("/local_mount/space/hypatia/1/users/jussi/speech/subjects.p", "rb"))

# original sample
#bad_subs = ['011202', '052402', '065203', '075901', '082802', '097601', '099302']

# equalized VIQs
#bad_subs = ['011202', '013703','052402', '065203', '075901', '082802', '097601', 
#            '099302', '052902', '090902','093101','063101','100001','076301','048102']

# equalized NVIQs
bad_subs = ['011202', '052402', '065203', '075901', '082802', '097601', '099302',
            '076301', '100001', '052901', '096603', '032902', '013703', '011201', 
            '048102', '052902', '075401']

for bad_sub in bad_subs:
    index = sub_info['sub_ID'].index(bad_sub)  
    [sub_info[key].pop(index) for key in sub_info.keys()]

n_asd = len([i for i in sub_info['ASD'] if i=='Yes'])
n_td = len([i for i in sub_info['ASD'] if i=='No'])

#%%
score_label = 'SRS_total'

scores_ASD = [val for ind,val in enumerate(sub_info[score_label]) if sub_info['ASD'][ind]=='Yes']
scores_TD = [val for ind,val in enumerate(sub_info[score_label])  if sub_info['ASD'][ind]=='No']

# indices of None, i.e. missing scores
null_inds_ASD = [ind for ind in range(len(scores_ASD)) if scores_ASD[ind]==None]
null_inds_ASD.reverse() # reverse so that largest are removed first
# remove Nones and the corresponding brain data value
for ind in null_inds_ASD:
    scores_ASD.pop(ind)
    
null_inds_TD = [ind for ind in range(len(scores_TD)) if scores_TD[ind]==None]
null_inds_TD.reverse() # reverse so that largest are removed first
# remove Nones and the corresponding brain data value
for ind in null_inds_TD:
    scores_TD.pop(ind)
    
#%%
mean_ASD = np.mean(scores_ASD)
std_ASD = np.std(scores_ASD)
min_ASD = min(scores_ASD)
max_ASD = max(scores_ASD)

print('\n%s ASD - mean: %f, stdev: %f, min: %f, max: %f' % (score_label, mean_ASD, 
                                                          std_ASD, min_ASD, max_ASD))

if scores_TD:
    mean_TD = np.mean(scores_TD)
    std_TD = np.std(scores_TD)
    min_TD = min(scores_TD)
    max_TD = max(scores_TD)

    print('%s TD - mean: %f, stdev: %f, min: %f, max: %f' % (score_label, mean_TD, 
                                                              std_TD, min_TD, max_TD))
    
    t,p = stats.ttest_ind(scores_ASD,scores_TD)
    print('Difference %s ASD vs TD: p=%f' % (score_label,p))

