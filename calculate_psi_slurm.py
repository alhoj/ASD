#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:00:25 2020

@author: ja151
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:51:58 2020

@author: ja151
"""

import mne
import sys
import pickle
import numpy as np
import os
from itertools import permutations

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

#%% subject ID from command line argument
sub_ID = sys.argv[1]
print('Calculating psi for subject %s' % sub_ID, flush=True)

#%%

# conditions
conds = ['MSS', 'SWS']
n_conds = len(conds)
equalize_event_counts = True
# include_baseline = False

# connectivity params
con_mode = 'cwt_morlet'

# times
tmin = -0.5 # None for beginning of epochs
tmax = None # None for end of epochs

# freqs
fmin = (4, 8, 12, 30, 60) # lower freqs
fmax = (8, 12, 30, 60, 100) # upper freqs
fstep = (0.5, 0.5, 2, 3, 4) # band-specific steps
cwt_freqs = []
for i,step in enumerate(fstep):
    cwt_freqs.append(np.arange(fmin[i], fmax[i]+step, step))
cwt_freqs = np.concatenate(cwt_freqs)
n_cycles = 7.

# rois
roi_names = ['AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
             'intrapariet_dSPM_10verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'intrapariet_dSPM_5verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'intrapariet_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
              'intrapariet_dSPM_10verts_MSS_SWS_peak0-500ms_auto-lh',
              'G_pariet_inf-Supramar_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
              'G_pariet_inf-Supramar_dSPM_5verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'S_postcentral_dSPM_5verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'S_postcentral_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
              'A1seedCon_wpli2_debiased_10verts_MSS-SWS_mean0-1500ms-lh',
              'A1seedCon_wpli2_debiased_10verts_MSS-SWS_mean0-1500ms_adjVerts-lh',
              'A1seedCon_wpli2_debiased_5verts_MSS-SWS_mean0-1500ms_adjVerts-lh',
              'A1seedCon_wpli2_debiased_5verts_MSS-SWS_mean0-1500ms-lh',
              'S_postcentral_dSPM_10verts_MSS_SWS_peak0-500ms_auto-lh',
              'S_postcentral_dSPM_10verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'S_postcentral_dSPM_5verts_MSS_SWS_mean0-1500ms_auto-lh',
              'S_postcentral_dSPM_5verts_MSS_SWS_mean0-1500ms_adjVerts_auto-lh',
              'S_postcentral_MNE_5verts_MSS_SWS_peak0-500ms_auto-lh',
              'S_postcentral_MNE_5verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'S_postcentral_MNE_5verts_MSS_SWS_mean0-1500ms_auto-lh',
              'S_postcentral_MNE_5verts_MSS_SWS_mean0-1500ms_adjVerts_auto-lh',
              'newIP_MNE_5verts_MSS_SWS_peak0-500ms_auto-lh',
              'newIP_MNE_5verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'S_postcentral_intraparietal_dSPM_5verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh',
              'S_postcentral_intraparietal_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
              'S_postcentral_intraparietal_dSPM_10verts_MSS_SWS_peak0-500ms_auto-lh',
              'S_postcentral_intraparietal_dSPM_10verts_MSS_SWS_peak0-500ms_adjVerts_auto-lh'
             ]
# calculate connectivity from the ROIs to the rest of the brain
# or between the ROIs only; can also be vector with seed ROIs 
# marked with 1's and targets with 0's
rois_as_seed = [1]+list(np.zeros(len(roi_names)-1, dtype=int))

# params for source estimates
snr = 1.0
lambda2 = 1.0 / snr ** 2
stc_method = 'MNE'
pick_ori = None #'normal'
tc_extract = 'pca_flip'

out_id = con_mode
overwrite = True
n_jobs = 1

#%% 
sub_path = '%s/%s/' % (paths['cluster'], sub_ID)    
inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                             % (sub_path, sub_ID))
src = inv['src']    
verts = [s['vertno'] for s in src]
n_verts = len(np.concatenate(verts))
fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]

epochs = mne.read_epochs('%s/%s_speech_1-100Hz_notch60Hz_-500-2000ms-epo.fif' 
                         % (sub_path, sub_ID), proj=False)
sfreq = epochs.info['sfreq']
times = epochs.times
if tmin is None: 
    tmin = times[0]
if tmax is None: 
    tmax = times[-1]

epochs.crop(tmin, tmax)
times = epochs.times

# if include_baseline:
#     epochs.baseline = None
#     epochs.shift_time(0, relative=False)

if equalize_event_counts:
    # check which conditions only have one stimulus
    conds1 = [cond for cond in conds if len(epochs[cond].event_id)==1]
    if conds1:
        epochs1 = [epochs[cond] for cond in conds1]
        # and which two
        conds2 = list(set(conds) - set(conds1))
        # equalize those with one first        
        print('Equalizing epoch counts for %s' % conds1)
        epochs.equalize_event_counts(conds1)
        if conds2:
            print('Dropping epochs from %s to match %s' % (conds2, conds1))
            # and then equalize those with two with the number of those with one
            n_epochs = len(epochs[conds1[0]])
            epochs2 = [epochs[cond][0:n_epochs] for cond in conds2]
            # combine back to one dict
            epochs = {cond: epochs for cond,epochs 
                      in zip(conds1 + conds2, epochs1 + epochs2)}
    else:
        epochs.equalize_event_counts(conds)   
n_epochs = len(epochs[conds[0]])

# output filename
fn_out = 'psi_%d-%dms_%d-%dHz_%s' % (int(tmin*1000), int(tmax*1000), 
                                     np.min(fmin), np.max(fmax), '-'.join(conds))
fn_out = fn_out.replace('/','')
if rois_as_seed and type(rois_as_seed)==bool:
    fn_out += '_seedROI'
else:
    fn_out += '_betweenROIs'    
if out_id:
    fn_out += '_%s' % out_id

out_path = '%s/p/%s.p' % (sub_path, fn_out)    
# check if output already exist
if os.path.exists(out_path) and not overwrite:
    print('Output already exists for %s' % sub_ID)
    exit()

rois = []
for roi_name in roi_names:
    roi_path = '%s/%s/rois/%s.label' % (paths['cluster'], sub_ID, 
                                        roi_name)
    roi_path_fsave = '%s/rois/%s.label' % (paths['cluster'], roi_name)
    if os.path.exists(roi_path):
        rois.append(mne.read_label(roi_path, subject=fs_id))
    elif os.path.exists(roi_path_fsave):
        roi = mne.read_label(roi_path_fsave, subject='fsaverageJA')
        rois.append(roi.copy().morph(subject_to=fs_id, 
                                     subject_from='fsaverageJA',
                                     grade=verts,
                                     subjects_dir=paths['fs']))
    else:
        rois.append(mne.read_labels_from_annot('fsaverageJA', 
                                                parc='PALS_B12_Lobes', 
                                                subjects_dir=paths['fs'], 
                                                regexp=roi_name)[0])
n_rois = len(rois) 
# if rois[0].subject == 'fsaverageJA':
#     temp = []
#     for roi in rois:
#         temp.append(roi.copy().morph(subject_to=fs_id, 
#                                     subject_from='fsaverageJA',
#                                     grade=verts,
#                                     subjects_dir=paths['fs']))
#     rois = temp
 

psi_dict = {}

#%%
for i_cond,cond in enumerate(conds):
    print('\nCondition: %s' % cond, flush=True)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs[cond], 
                                                 inverse_operator=inv, 
                                                 lambda2=lambda2, 
                                                 method=stc_method, 
                                                 pick_ori=pick_ori,
                                                 return_generator=False) 

    roi_tcs = mne.extract_label_time_course(stcs, rois, src, tc_extract,
                                                return_generator=False) 
    
    if rois_as_seed and type(rois_as_seed)==bool:
        tcs = list(zip(roi_tcs, stcs))
        n_signals = 1 + len(np.concatenate(verts))
        inds = mne.connectivity.seed_target_indices([0], np.arange(1, n_signals))
        con_labels = []
    else:
        tcs = roi_tcs
        if type(rois_as_seed)==bool:
            pairs = permutations(np.arange(n_rois), 2)
            seeds, targets, con_labels = [], [], []
            for pair in pairs:
                seeds.append(pair[0])
                targets.append(pair[1])
                con_labels.append('%s -> %s' % (roi_names[pair[0]], 
                                             roi_names[pair[1]]))
            inds = (np.array(seeds), np.array(targets))
        else:
            inds = mne.connectivity.seed_target_indices(
                np.where(np.array(rois_as_seed)==1)[0], 
                np.where(np.array(rois_as_seed)==0)[0])
            con_labels = []
        
    if con_mode=='cwt_morlet':
        psi, freqs, _, n_epochs, _ = mne.connectivity.phase_slope_index(
            tcs, mode=con_mode, indices=inds, sfreq=sfreq, tmin=tmin,
            tmax=tmax, fmin=fmin, fmax=fmax, cwt_freqs=cwt_freqs, 
            cwt_n_cycles=n_cycles, n_jobs=12)
    elif con_mode=='multitaper':
        psi, freqs, _, _, _ = mne.connectivity.phase_slope_index(
            tcs, mode=con_mode, indices=inds, sfreq=sfreq, tmin=0.,
            tmax=tmax, fmin=fmin, fmax=fmax, n_jobs=1)
    
    psi_dict.update({cond: psi.squeeze()})

    
    data_struct = {'psi': psi_dict, 'conds': conds, 'rois': roi_names, 
                   'times': times, 'freqs': freqs, 'fmin': fmin, 'fmax': fmax,
                   'n_epochs': n_epochs, 'n_cycles': n_cycles, 
                   'stc_method': stc_method, 'lambda2': lambda2, 
                   'pick_ori': pick_ori, 'con_mode': con_mode, 
                   'rois_as_seed': rois_as_seed, 'con_labels': con_labels}

#% save connectivity data
print('Saving %s' % out_path)
f = open(out_path, 'wb')
pickle.dump(data_struct, f)
f.close()
