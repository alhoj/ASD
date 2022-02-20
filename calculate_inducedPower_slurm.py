#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:51:58 2020

@author: ja151
"""

import mne
from MNEprepro import MNEprepro
import pickle
import numpy as np
import random
import sys
import multiprocessing
import os
import matplotlib.pyplot as plt

#%% subject ID from command line argument
sub_ID = sys.argv[1]
print('Calculating entrainment for subject %s' % sub_ID, flush=True)

#%% set up
paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# which conditions to calculate PAC for
conds = ['MSS', 'SWS']
n_conds = len(conds)

# times
tmin = None # None for beginning of epochs
tmax = None # None for end of epochs

# freqs
fmin = 4
fmax = 100
fstep = 0.5
freqs = np.arange(fmin, fmax+fstep, fstep)
n_freqs = len(freqs)
n_cycles = freqs / 2.
# n_cycles = 7

# brain regions
roi_names = ['AC_MSS_SWS_manual-lh',
             'IP_MSS_SWS_manual-lh',
             'IF_MSS_SWS_manual-lh']
 
n_rois = len(roi_names)

# params for source estimates
snr = 1.0
lambda2 = 1.0 / snr ** 2
stc_method = 'MNE'
pick_ori = 'normal'
baseline_mode = 'zscore'
postfix = 'dSPMfROIsManual0-500ms'

overwrite = True

# number of cores (for parallel processing)
num_jobs = 1 # multiprocessing.cpu_count()

#%% load data etc.

# subject path
sub_path = '%s/%s/' % (paths['cluster'], sub_ID)
if sub_info['ASD'][sub_info['sub_ID'].index(sub_ID)]=='Yes':
    asd_label = 'ASD'
else:
    asd_label = 'TD'

epochs = mne.read_epochs('%s/%s_speech_0-100Hz_-200-2000ms_noReject-epo.fif' 
                         % (sub_path, sub_ID), proj=False)
epochs.equalize_event_counts(conds)

sfreq = epochs.info['sfreq']
if tmin: 
    i_tmin = int((tmin-epochs.times[0])*sfreq)
else: 
    i_tmin = 0
    tmin = epochs.times[i_tmin]
if tmax: 
    i_tmax = int((tmax-epochs.times[0])*sfreq)
else: 
    i_tmax = -1
    tmax = round(epochs.times[i_tmax])
times = epochs.times[i_tmin:i_tmax]
    
# read inverse operator
inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                             % (sub_path, sub_ID))
src = inv['src']
verts = [s['vertno'] for s in src]
n_verts = len(np.concatenate(verts))
fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]

# output filename
fn_out = 'induced_%d-%dms_%d-%dHz_%s_%s' % (int(tmin*1000), int(tmax*1000), 
                                             fmin, fmax, '-'.join(conds).replace('/',''),
                                             postfix)
# check if output already exist
if os.path.exists('%s/%s.p' % (sub_path, fn_out)) and not overwrite:
    print('Output files already exists for %s' % sub_ID)
    exit()

#%%
power_conds = {}
for cond in conds:
    print('\nCalculating induced power', flush=True)
    epochs_induced = epochs[cond].copy().subtract_evoked()    
            
    power_rois = []
    for i_roi,roi_name in enumerate(roi_names):
        roi = mne.read_label('%s/rois/%s.label' % (sub_path, roi_name),
                             subject=fs_id)
        
        power,_ = mne.minimum_norm.source_induced_power(epochs_induced, inv, 
                                                        freqs, roi, lambda2, 
                                                        method=stc_method, 
                                                        n_cycles=n_cycles, 
                                                        pick_ori=pick_ori,
                                                        baseline=(None, 0),
                                                        baseline_mode=baseline_mode, 
                                                        n_jobs=num_jobs)
        power_rois.append(power.mean(0))
        
    power_conds.update({cond: np.array(power_rois)[:,:,i_tmin:i_tmax].squeeze()})

data_struct = {'induced_power': power_conds, 'conds': conds, 'rois': roi_names, 
               'times': times, 'freqs': freqs, 'n_cycles': n_cycles, 
               'method': stc_method, 'lambda2': lambda2, 'pick_ori': pick_ori,
               'baseline_mode': baseline_mode}

#% save ITPC/PDF data
save_path = '%s/p/%s.p' % (sub_path, fn_out)
print('Saving %s' % save_path)
f = open(save_path, 'wb')
pickle.dump(data_struct, f)
f.close()
    
print('All done!')
    
        
