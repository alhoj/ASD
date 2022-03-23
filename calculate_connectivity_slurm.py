#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:00:25 2020

@author: ja151
"""

import sys
sys.path.append('/autofs/cluster/transcend/jussi/scripts')
import mne
import pickle
import numpy as np
import os
import config_connectivity as cfg
from helper_functions import module_to_dict

paths = cfg.paths

# subject ID from command line argument
sub_ID = sys.argv[1]
print('Calculating connectivity for subject %s' % sub_ID, flush=True)

# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# exclude subjects
for sub in cfg.exclude:
    ind = sub_info['sub_ID'].index(sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
     if len(sub_info[key])==n_subs]
sub_IDs = sub_info['sub_ID']

if cfg.rois_as_seed:
    # read the source space we are morphing to
    src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
    src = mne.read_source_spaces(src_fname)
    verts_fsave = [s['vertno'] for s in src]
    n_verts_fsave = len(np.concatenate(verts_fsave))

#%% 

sub_path = '%s/%s/' % (paths['cluster'], sub_ID)    
inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                             % (sub_path, sub_ID))
src = inv['src']    
verts = [s['vertno'] for s in src]
fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
epochs = mne.read_epochs('%s/%s_speech_1-100Hz_notch60Hz_-500-2000ms-epo.fif' 
                         % (sub_path, sub_ID), proj=False)
sfreq = epochs.info['sfreq']
times = epochs.times

# times
tmin = cfg.tmin
tmax = cfg.tmax
if tmin is not None: 
    i_tmin = list(times).index(tmin)
else: 
    i_tmin = 0
    tmin = times[i_tmin]
if tmax is not None: 
    i_tmax = list(times).index(tmax)
else: 
    i_tmax = -1
    tmax = round(times[i_tmax])
    
# frequencies
fmin = cfg.fmin
fmax = cfg.fmax
fstep = cfg.fstep
freqs = np.arange(fmin, fmax+fstep, fstep)

conds = cfg.conds
if cfg.equalize_epoch_counts:
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

if cfg.add_baseline_cond: 
    epochs_bl = epochs[0:n_epochs]
    i_tmin_bl = 0
    i_tmax_bl = list(times).index(0)
    this_conds = conds + ['Baseline']
else:
    this_conds = conds

rois = []
for roi_name in cfg.rois:
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


#%%
if cfg.rois_as_seed:
    for roi in rois:
        print('\nCalculating connectivity from %s to the rest of the brain\n' % roi.name)
        roi_label = roi.name#'%s-%s' % (roi.name[0:2], roi.name[-2:])
        # output filename
        fn_out = 'con_%d-%dms_%d-%dHz_%s' % (int(tmin*1000), int(tmax*1000), 
                                             fmin, fmax, '-'.join(conds))
        fn_out = fn_out.replace('/','')
        fn_out += '_seedROI_%s' % roi_label
        fn_out += '_%s' % '-'.join(cfg.con_method)
        out_path = '%s/p/%s.p' % (sub_path, fn_out)    
        # check if output already exist
        if os.path.exists(out_path) and not cfg.overwrite:
            print('Output already exists for %s' % sub_ID)
            continue
        
        cons_dict = {}
        for cond in conds:
            print('\nCondition: %s' % cond, flush=True)
            if cond=='Baseline':
                this_epochs = epochs_bl
            else:
                this_epochs = epochs[cond]
            stcs = mne.minimum_norm.apply_inverse_epochs(this_epochs, 
                                                         inverse_operator=inv, 
                                                         lambda2=cfg.lambda2, 
                                                         method=cfg.stc_method, 
                                                         pick_ori=cfg.pick_ori,
                                                         return_generator=False)
            
            roi_tcs = mne.extract_label_time_course(stcs, [roi], src, cfg.tc_extract,
                                                    return_generator=False)

            tcs = list(zip(roi_tcs, stcs))
            n_signals = 1 + len(np.concatenate(verts))
            inds = mne.connectivity.seed_target_indices([0], np.arange(1, n_signals))
                
            con, freqs, _, n_epochs, _ = mne.connectivity.spectral_connectivity(
                tcs, method=cfg.con_method, mode=cfg.con_mode, indices=inds, 
                sfreq=sfreq, cwt_freqs=freqs, cwt_n_cycles=cfg.n_cycles, n_jobs=cfg.n_jobs)
            
            if cfg.average_times:
                if cond=='Baseline':
                    i_tmin = i_tmin_bl
                    i_tmax = i_tmax_bl
                if type(con)==list:
                    con = [this_con[:,:,i_tmin:i_tmax].mean(-1) for this_con in con]
                else:
                    con = con[:,:,i_tmin:i_tmax].mean(-1)
        
            if cfg.average_freqs:
                if type(con)==list:
                    con = [this_con.mean(1) for this_con in con]
                else:
                    con = con.mean(1)
            
            cons_dict.update({cond: con})
            # n_epochs_dict.update({cond: n_epochs})
        
        
else:
    # output filename
    fn_out = 'con_%d-%dms_%d-%dHz_%s' % (int(tmin*1000), int(tmax*1000), 
                                            fmin, fmax, '-'.join(conds))
    fn_out = fn_out.replace('/','')
    fn_out += '_betweenROIs'    
    fn_out += '_%s' % '-'.join(cfg.con_method)
    out_path = '%s/p/%s.p' % (sub_path, fn_out)    
    # check if output already exist
    if os.path.exists(out_path) and not cfg.overwrite:
        print('Output already exists for %s' % sub_ID)
        exit()
    
    cons_dict = {}
    for cond in conds:
        print('\nCondition: %s' % cond, flush=True)
        if cond=='Baseline':
            this_epochs = epochs_bl
        else:
            this_epochs = epochs[cond]
        stcs = mne.minimum_norm.apply_inverse_epochs(this_epochs, 
                                                     inverse_operator=inv, 
                                                     lambda2=cfg.lambda2, 
                                                     method=cfg.stc_method, 
                                                     pick_ori=cfg.pick_ori,
                                                     return_generator=False)
        
        roi_tcs = mne.extract_label_time_course(stcs, rois, src, cfg.tc_extract,
                                                return_generator=False)
           
        con, freqs, _, n_epochs, _ = mne.connectivity.spectral_connectivity(
            roi_tcs, method=cfg.con_method, mode=cfg.con_mode, indices=None, 
            sfreq=sfreq, cwt_freqs=freqs, cwt_n_cycles=cfg.n_cycles, n_jobs=cfg.n_jobs)
        
        if cfg.average_times:
            if type(con)==list:
                con = [this_con[:,:,:,i_tmin:i_tmax].mean(-1) for this_con in con]
            else:
                con = con[:,:,:,i_tmin:i_tmax].mean(-1)
    
        if cfg.average_freqs:
            if type(con)==list:
                con = [this_con.mean(-1) for this_con in con]
            else:
                con = con.mean(1)
        
        cons_dict.update({cond: con})
        # n_epochs_dict.update({cond: n_epochs})
    
# convert cfg module to dict
cfg_dict = module_to_dict(cfg)
# output struct
out = {'cfg': cfg_dict, 'cons': cons_dict, 'conds': this_conds, 
       'times': (tmin, tmax), 'freqs': freqs, 'n_epochs': n_epochs}

#% save connectivity data
print('Saving %s' % out_path)
f = open(out_path, 'wb')
pickle.dump(out, f)
f.close()
                
