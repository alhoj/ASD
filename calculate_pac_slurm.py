#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:05:49 2021

@author: ja151
"""

import sys
import mne
import numpy as np
import pickle
import os
import scipy
import multiprocessing
import random

if '/autofs/cluster/transcend/jussi/tools/' not in sys.path:
    sys.path.append('/autofs/cluster/transcend/jussi/tools/')
from helper_functions import DAR_MI, canolty_MI, ozkurt_MI, module_to_dict
from pactools.utils.parallel import Parallel, delayed
from pactools import multiple_band_pass
from pactools.dar_model.preprocess import multiple_extract_driver
import config_pac as cfg

#%% subject ID from command line argument
sub_ID = sys.argv[1]
print('Calculating phase-amplitude coupling (PAC) for subject %s' % sub_ID)

# paths
paths = cfg.paths

# subject path
sub_path = '%s/%s/' % (paths['cluster'], sub_ID)

# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()


#%% load data etc.

if cfg.phase_fq_min=='PAF':
    phase_fq_label = 'PAF'
    phase_fq_min = sub_info['PAF'][sub_info['sub_ID'].index(sub_ID)]
    phase_fq_max = sub_info['PAF'][sub_info['sub_ID'].index(sub_ID)]
    print('Using individual peak alpha frequency (%.2f Hz) for phase' % phase_fq_min)
else:
    phase_fq_min = cfg.phase_fq_min
    phase_fq_max = cfg.phase_fq_max
    phase_fq_step = cfg.phase_fq_step
    phase_fq_label = '%d-%dHz' % (phase_fq_min, phase_fq_max)
    print('Using %.1f-%.1fHz with %.1fHz step for phase' % (phase_fq_min, 
                                                      phase_fq_max, 
                                                      phase_fq_step))
amp_fq_min = cfg.amp_fq_min
amp_fq_max = cfg.amp_fq_max
amp_fq_step = cfg.amp_fq_step
amp_fq_label = '%d-%dHz' % (amp_fq_min, amp_fq_max)
print('Using %d-%dHz with %dHz step for amplitude' % (amp_fq_min, amp_fq_max, 
                                                      amp_fq_step))

# exit if no PAF exists for the subject
if not phase_fq_min:
    print('No peak alpha frequency found, exiting')
    exit()

phase_fq_range = np.arange(phase_fq_min,phase_fq_max+phase_fq_step,phase_fq_step)
amp_fq_range = np.arange(amp_fq_min,amp_fq_max+amp_fq_step,amp_fq_step)
n_phase_fq = len(phase_fq_range)
n_amp_fq = len(amp_fq_range)
phase_fq_width = cfg.phase_fq_width
amp_fq_width = phase_fq_max * 2 # should be at least twice the highest freq of the phase range

# read inverse operator
inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                             % (sub_path, sub_ID))
src = inv['src']
verts = [s['vertno'] for s in src]
fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]

# find raw data and events
raw_names = [f for f in os.listdir(sub_path) if f.endswith('raw_tsss.fif')]
raw_names.sort()
events_names = ['%s-eve.fif' % os.path.splitext(raw_name)[0] for raw_name in raw_names]
 
# read raws
raws = [mne.io.read_raw_fif('%s/%s' % (sub_path, raw_name), preload=False) 
        for raw_name in raw_names]   
# read events
events = [mne.read_events('%s/%s' % (sub_path, events_name)) 
          for events_name in events_names]  
# concatenate raws and events    
raw, events = mne.concatenate_raws(raws, events_list=events)
# get sampling frequency
sfq = raw.info['sfreq']

# set event IDs
event_IDs = np.unique(events[:,2])                
event_IDs_dict = {'Speech/A': event_IDs[0], 'Speech/B': event_IDs[1],
                  'Jabber/A': event_IDs[2], 'Jabber/B': event_IDs[3],
                  'MSS/B': event_IDs[4], 'SWS/B': event_IDs[5],
                  'Noise/A': event_IDs[6], 'Noise/B': event_IDs[7]}

# list of stimuli corresponding to conditions
stims = [key for key in event_IDs_dict.keys() for cond in cfg.conds if cond in key]

# drop events that occurred after raw data acquisition was stopped
events = events[events[:,0]<int(raw.n_times-(cfg.tmax*sfq))]
# drop events whose baseline would go to negative samples
events = events[events[:,0]>int(cfg.baseline_length*sfq)]

events_list = [events[events[:,2]==event_IDs_dict[stim]][:,0] 
               for stim in stims]

if cfg.equalize_epoch_counts=='stim':
    # equalize epoch counts across all stimuli
    n_epochs = np.min([events.shape[0] for events in events_list])
    events_list = [events[0:n_epochs] for events in events_list]

# put events in dict with stimulus labels
events_stim = {stim: events for stim,events in 
               zip(stims, events_list)}

# combine stimulus events within condition
events_cond = {}
for cond in cfg.conds:
    events_cond[cond] = np.sort(np.hstack([events for stim,events 
                                           in events_stim.items() 
                                           if cond in stim]))  
if cfg.equalize_epoch_counts=='cond':
    # equalize epoch counts across conditions
    n_epochs = np.min([len(events) for events in events_cond.values()])
    events_cond = {cond:events[0:n_epochs] for cond,events in events_cond.items()}
        

if cfg.add_baseline: # add baseline condition
    epoch_length = cfg.tmax-cfg.tmin
    multiplier = int(epoch_length / cfg.baseline_length)
    n_epochs_bl = n_epochs*multiplier
    conds = cfg.conds + ['Baseline']
    events_cond['Baseline'] = np.asarray(np.sort(random.sample(list(events[:,0]), 
                                                               n_epochs_bl)))    

#%% read ROI labels etc

# for hemi in hemis:
    
# print('\nHemisphere: %s' % hemi)
if cfg.vertexwise:
    spat_label = 'vertexwise_%s' % '-'.join(cfg.roi_labels)
elif  len(cfg.roi_labels)==1:
    spat_label = cfg.roi_labels[0]
else:
    spat_label = '%dROIs' % len(cfg.roi_labels)
    
# output identifier
out_ID = '%s_%s_%s_%s_%d-%dms_%s' % (spat_label, ('-').join(conds).replace('/',''), 
                                     phase_fq_label, amp_fq_label, 
                                     int(cfg.tmin*1000), int(cfg.tmax*1000),
                                     cfg.pac_method)
if cfg.concatenate_epochs:
    out_ID += '_concatEpochs'
    
if cfg.postfix:
    out_ID += '_%s' % cfg.postfix

if os.path.exists('%s/pac_%s.p' % (sub_path, out_ID)) and not cfg.overwrite:
    print('%s already exists for %s' % (out_ID, sub_ID))
    exit()

# read ROIs
rois = []
for roi_label in cfg.roi_labels:
    path = '%s/%s/rois/%s.label' % (paths['cluster'], sub_ID, 
                                    roi_label)
    path_alt = '%s/rois/%s.label' % (paths['cluster'], roi_label)
    if os.path.exists(path):
        rois.append(mne.read_label(path, subject=fs_id))
    elif os.path.exists(path_alt):
        label = mne.read_label(path_alt, subject='fsaverageJA')
        label_morphed = label.copy().morph(subject_to=fs_id, 
                                    subject_from='fsaverageJA',
                                    grade=verts, subjects_dir=paths['fs'])
        rois.append(label_morphed)
    else:
        rois.append(mne.read_labels_from_annot('fsaverageJA', 
                                                parc='PALS_B12_Lobes', 
                                                subjects_dir=paths['fs'], 
                                                regexp=roi_label)[0])
      
if cfg.vertexwise:
    # all ROIs mask
    mask = rois[0]
    for roi in rois[1::]: mask += roi
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, method=cfg.stc_method, 
                                             lambda2=cfg.lambda2, 
                                             pick_ori=cfg.pick_ori,
                                             label=mask)
else:
    mask = None
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, method=cfg.stc_method, 
                                             lambda2=cfg.lambda2, pick_ori=cfg.pick_ori)
    
    
if cfg.vertexwise:
    # vertex time courses
    tcs = stc.data
else:
    # ROI time courses
    tcs = mne.extract_label_time_course(stc, rois, src, mode=cfg.tc_extract,
                                        return_generator=False)
    
# number of vertices/ROIs and time points
n_spats = tcs.shape[0]
n_times = tcs.shape[1]

del stc
#%%
if cfg.pac_method=='duprelatour':
    print('\nCalculating PAC using ' +
          'driven auto-regressive modeling by Dupre la Tour et al. (2017)',
          flush=True)
    
    pac_cond = {}
    for cond, events in events_cond.items(): 
        print('\n%s' % cond, flush=True) 
    
        if cfg.concatenate_epochs:
            MI = np.zeros((n_spats, n_phase_fq, n_amp_fq, 1))
        else:
            MI = np.zeros((n_spats, n_phase_fq, n_amp_fq, n_epochs))
        
        print('Extracting drivers...', flush=True)
        for i_phase,sigs in enumerate(multiple_extract_driver(tcs, sfq, phase_fq_range)):
            print('Driver %.1fHz' % phase_fq_range[i_phase], flush=True)
            filtered_phase, filtered_amp, filtered_phase_imag = sigs
                                                               
            # epoching
            if cond == 'Baseline':            
                this_tmin = -cfg.baseline_length
                this_tmax = 0
                n_epochs = n_epochs_bl
            else:
                this_tmin = cfg.tmin
                this_tmax = cfg.tmax                
            n_times_epoch = int((this_tmax - this_tmin) * sfq)            
            this_filtered_phase = np.zeros((n_spats, n_epochs, n_times_epoch))
            this_filtered_amp = np.zeros((n_spats, n_epochs, n_times_epoch))
            this_filtered_phase_imag = np.zeros((n_spats, n_epochs, n_times_epoch))
            for i,event in enumerate(events):
                start = int(event + this_tmin * sfq)
                stop = int(event + this_tmax * sfq)
                this_filtered_phase[:,i] = filtered_phase[:,start:stop]
                this_filtered_amp[:,i] = filtered_amp[:,start:stop]
                this_filtered_phase_imag[:,i] = filtered_phase_imag[:,start:stop]
                
            if cfg.concatenate_epochs:
                new_shape = (n_spats, 1, n_epochs*n_times_epoch)
                this_filtered_phase = this_filtered_phase.reshape(new_shape)
                this_filtered_amp = this_filtered_amp.reshape(new_shape)
                this_filtered_phase_imag = this_filtered_phase_imag.reshape(new_shape)
                
            this_filtered_amp /= np.std(this_filtered_amp)
                    
            MI[:,i_phase] = Parallel(n_jobs=cfg.num_jobs)\
                            (delayed(DAR_MI)(this_filtered_phase[i], 
                                             this_filtered_phase_imag[i],
                                             this_filtered_amp[i],
                                             amp_fq_range, sfq) 
                             for i in range(n_spats))
        
        # put MIs in dict with stimulus label
        pac_cond.update({cond: MI})
            
else: # use standard PAC metrics
    if cfg.pac_method=='canolty':
        print('\nCalculating PAC using method by ' +
              'Canolty et al. (2006)', flush=True)
        func = canolty_MI
    elif cfg.pac_method=='ozkurt':
        print('\nCalculating PAC using method by ' +
              'Ozkurt et al. (2011)', flush=True)
        func = ozkurt_MI
            
    print('\nFilter and Hilbert...', flush=True)
    filtered_phase = multiple_band_pass(tcs, sfq, phase_fq_range, 
                                        phase_fq_width, filter_method=cfg.filter_method)
    filtered_amp = multiple_band_pass(tcs, sfq, amp_fq_range, 
                                       amp_fq_width, filter_method=cfg.filter_method) 
    
    pac_cond = {}
    for cond, events in events_cond.items(): 
        print('\nCondition: %s' % cond, flush=True)
        
        # epoching
        if cond == 'Baseline':            
            this_tmin = -cfg.baseline_length
            this_tmax = 0
            n_epochs = n_epochs_bl
        else:
            this_tmin = cfg.tmin
            this_tmax = cfg.tmax       
        n_times_epoch = int((this_tmax - this_tmin) * sfq)         
        this_filtered_phase = np.zeros((n_phase_fq, n_spats, n_epochs, n_times_epoch), dtype=complex)
        this_filtered_amp = np.zeros((n_amp_fq, n_spats, n_epochs, n_times_epoch), dtype=complex)
        for i,event in enumerate(events):
            start = int(event + this_tmin * sfq)
            stop = int(event + this_tmax * sfq)
            this_filtered_phase[:,:,i] = filtered_phase[:,:,start:stop]
            this_filtered_amp[:,:,i] = filtered_amp[:,:,start:stop]
            
        if cfg.concatenate_epochs:
            new_shape = (n_phase_fq, n_spats, 1, n_epochs*n_times_epoch)
            this_filtered_phase = this_filtered_phase.reshape(new_shape)
            new_shape = (n_amp_fq, n_spats, 1, n_epochs*n_times_epoch)
            this_filtered_amp = this_filtered_amp.reshape(new_shape)

        # phase of the low frequency signals
        this_filtered_phase = np.angle(this_filtered_phase)        
        # amplitude of the high frequency signals
        this_filtered_amp = np.real(np.abs(this_filtered_amp))
            
        # normalization if ozkurt
        if cfg.pac_method == 'ozkurt':
            shape_high = this_filtered_amp.shape
            if cfg.concatenate_epochs:
                this_filtered_amp = this_filtered_amp.reshape(n_amp_fq*n_spats, n_epochs*n_times_epoch)
                norms = np.zeros(n_amp_fq*n_spats)
            else:
                this_filtered_amp = this_filtered_amp.reshape(n_amp_fq*n_spats*n_epochs, n_times_epoch)
                norms = np.zeros(n_amp_fq*n_spats*n_epochs)
            for i in range(len(norms)):
                # Euclidean norm when x is a vector, Frobenius norm when x
                # is a matrix (2-d array). More precise than sqrt(squared_norm(x)).
                nrm2, = scipy.linalg.get_blas_funcs(['nrm2'], [this_filtered_amp[i]])
                norms[i] = nrm2(this_filtered_amp[i])                    
            this_filtered_amp = this_filtered_amp.reshape(shape_high)
        
        
        # calculate modulation index for each ROI and frequency (and epoch)                        
        MI = Parallel(n_jobs=cfg.n_jobs)(delayed(func)(this_filtered_phase[:,i], 
                                                      this_filtered_amp[:,i]) 
                                        for i in range(n_spats))
                                        
        # put MIs in dict with stimulus label
        pac_cond.update({cond: MI})
    del filtered_phase, filtered_amp
        
# convert cfg module to dict
cfg_dict = module_to_dict(cfg)
# output data in dict
out = {'cfg': cfg_dict, 'pac': pac_cond, 'n_spats': n_spats,
       'times': (cfg.tmin, cfg.tmax), 'mask': mask,
       'n_epochs': n_epochs, 'phase_fq_range': phase_fq_range, 
       'amp_fq_range': amp_fq_range,} # 

# save PAC data
save_path = '%s/p/pac_%s.p' % (sub_path, out_ID)
print('\nSaving %s' % save_path)
f = open(save_path, 'wb')
pickle.dump(out, f)
f.close()
        
print('All done!', flush=True)

