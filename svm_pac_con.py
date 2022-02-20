#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:11:02 2021

@author: ja151
"""

import pickle
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
import seaborn as sns
import scipy.io
from sklearn import svm, preprocessing
from sklearn.model_selection import cross_validate

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'local': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

f = open('%s/p/subjects.p' % paths['local'], 'rb')
sub_info = pickle.load(f)
f.close()

## define bad subjects
# bad_subs = ['105801', '107301']
## equalized NVIQs - new sample N_ATD=29, N_ASD=29
# bad_subs = ['105801', '107301', '052902', '090902', '048102']
## equalized NVIQs - new sample N_ATD=28, N_ASD=28
bad_subs = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for bad_sub in bad_subs:
    ind = sub_info['sub_ID'].index(bad_sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
     if len(sub_info[key])==n_subs]
sub_IDs = sub_info['sub_ID']
exclude = [] # '098101', '098501', '030801'

fns = [
       'granger_N56_0.0-0.5s_8-12Hz_MSS-SWS_AC-con_phte-8.mat',
       'seedAC_wpli2_debiased_N56_8-12Hz_MSS-SWS_cdt0.01_pht1e-08_noReg.npy',
       'pac_ACseedConCluster_N56_MSS-SWS_phase8-12Hz_amp30-60Hz_cdt0.01_noReg.npy'
       ]


#% Read data
X = []
for fn in fns:
    if '.mat' in fn:
        data = scipy.io.loadmat('%s/npy/%s' % (paths['cluster'], fn))
        data = data['brains'].squeeze()
    else:
        data = np.load('%s/npy/%s' % (paths['cluster'], fn))
            
    inds_ASD = [i for i in range(len(sub_IDs)) if sub_info['ASD'][i]=='Yes']
    inds_TD = [i for i in range(len(sub_IDs)) if sub_info['ASD'][i]=='No']  
    if exclude:
        inds_exclude = [sub_IDs.index(sub_ID) for sub_ID in exclude]
        inds_exclude = list(np.sort(inds_exclude))
        inds_exclude.reverse() # reverse so that largest index is removed first
        # remove Nones and the corresponding brain data
        for i in inds_exclude:
            if sub_info['ASD'][i]=='Yes':
                i_exclude = inds_ASD.index(i)
                inds_ASD.pop(i_exclude)
            else:
                i_exclude = inds_TD.index(i)
                inds_TD.pop(i_exclude)
            data = np.delete(data, i_exclude)
        
    
    X.append(data)
    
X = np.array(X).T
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
y = [0 if asd=='No' else 1 for asd in sub_info['ASD']]
    
#% svm classification

clf = svm.SVC(kernel='linear', C=1, random_state=7)
# clf = svm.SVC(kernel='rbf', gamma=0.7, C=1, random_state=7)
scoring = ['balanced_accuracy', 'recall']
scores = cross_validate(clf, X, y, scoring=scoring, cv=5)
balanced_acc = np.mean(scores['test_balanced_accuracy'])
sensitivity = np.mean(scores['test_recall'])
specificity = 2*balanced_acc-sensitivity

print('Balanced accuracy: %f; sensitivity: %f; specificity: %f' % (balanced_acc, 
                                                                   sensitivity,
                                                                   specificity))