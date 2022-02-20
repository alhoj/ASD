#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:51:58 2020

@author: ja151
"""

import mne
import pickle
import numpy as np
import random
import sys
import multiprocessing
import os

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

# subject path
sub_path = '%s/%s/' % (paths['cluster'], sub_ID)

# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# which conditions to calculate PAC for
conds = ['Speech/B', 'Jabber/B', 'MSS', 'SWS', 'Noise/B']
n_conds = len(conds)

# times
# times
tmin = None # None for beginning of epochs
tmax = None # None for end of epochs
decim = 10 # temporal decimation factor

# freqs
fmin = 1
fmax = 20
fstep = 0.5
freqs = np.arange(fmin, fmax+fstep, fstep)
n_freqs = len(freqs)
n_cycles = freqs / 2.

# brain regions
roi_labels = [None] # None for whole brain
n_rois = len(roi_labels)
parc = 'aparc' # None for reading ROIs from /autofs/cluster/transcend/jussi/rois/
vertexwise = True # True -> use rois as mask and calculate PAC for each vertex 
                  # False -> average time courses within each ROI

# params for source estimates
snr = 1.0
lambda2 = 1.0 / snr ** 2
stc_method = 'MNE'
pick_ori = 'normal'
tc_extract = 'mean_flip' # only if vertexwise == False

overwrite = False

# number of cores (for parallel processing)
num_jobs = 1 # multiprocessing.cpu_count()

#%% load data etc.

epochs = mne.read_epochs('%s/%s_speech_0-40Hz_-200-2000ms_noReject-epo.fif' 
                         % (sub_path, sub_ID), proj=False)
times = epochs.times
times = times[0:-1:decim]
sfreq = epochs.info['sfreq']
if tmin: 
    i_tmin = int((tmin-times[0])*sfreq)
else: 
    i_tmin = 0
    tmin = times[i_tmin]
if tmax: 
    i_tmax = int((tmax-times[0])*sfreq)
else: 
    i_tmax = -1
    tmax = round(times[i_tmax])

# output filename
fn_out = 'itpc_pdf_%d-%dms_%d-%dHz_%s' % (int(tmin*1000), int(tmax*1000), 
                                          fmin, fmax, '-'.join(conds).replace('/',''))
# check if output already exist
if os.path.exists('%s/%s.p' % (sub_path, fn_out)) and \
    os.path.exists('%s/pdf_peak_coh_%s-lh.stc' % (sub_path, conds[0])) \
    and not overwrite:
    print('Output files already exists for %s' % sub_ID)
    exit()

# read inverse operator
inv = mne.minimum_norm.read_inverse_operator('%s/%s_speech-oct6-inv.fif' 
                                             % (sub_path, sub_ID))
src = inv['src']
verts = [s['vertno'] for s in src]
n_verts = len(np.concatenate(verts))
fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]

# list of stimuli corresponding to conditions
stims = [key for key in epochs.event_id.keys() for cond in conds if cond in key]
n_stims = len(stims)
# within-stimulus epochs
epochs_within = []
for i_stim,stim in enumerate(stims):
    epochs_within.append(epochs[stim])
mne.epochs.equalize_epoch_counts(epochs_within)
n_epochs = len(epochs_within[0])
    
# across-stimulus epochs, picked randomly from each stimulus group
epochs_across = []
inds_epoch = np.arange(n_epochs)
random.shuffle(inds_epoch) # shuffle epoch indices
rand_parts = np.array_split(inds_epoch,n_stims) # random partitions of epoch_inds
rand_parts = [np.sort(rand_part) for rand_part in rand_parts] # sort to chronological order    
inds_rep = np.arange(n_stims) # repetition indices for the across-stim condition
for i_stim in np.arange(n_stims):
    for i_epochs,epochs in enumerate(epochs_within):
        if i_epochs==0: inds_rep = np.roll(inds_rep,1) # this is to balance the count between
                                                        # epochs from different stimuli in case 
                                                         # n_epochs is not divisible with n_stims
        epochs_across.append(epochs[rand_parts[inds_rep[i_epochs]]])
    # combine the epochs into one across-stimulus category
    epochs_across[i_stim::] = [mne.concatenate_epochs(epochs_across[i_stim::])]

epochs_list = epochs_within + epochs_across
    
#% Calculate inter-trial phase coherence and phase dissimilarity function

print('\nCalculating inter-trial phase coherence (ITPC)', flush=True)
if vertexwise:
    n_spats = n_verts
else:
    n_spats = n_rois
# itpc_stims = np.zeros((len(epochs_list), n_spats, n_freqs))

itpc_labels = []
itpcs = {}
for i_epochs,epochs in enumerate(epochs_list):
    if i_epochs>len(epochs_within)-1:
        itpc_labels.append('across-%s_%d' % (('-').join(list(epochs.event_id.keys())).replace('/',''), 
                                             i_epochs-(len(epochs_within)-1)))
    else:
        itpc_labels.append('within-%s' % ('-').join(list(epochs.event_id.keys())).replace('/',''))      
        
    print('\nITPC type: %s' % itpc_labels[i_epochs])
        
    for i_roi,roi_label in enumerate(roi_labels):
        if parc and roi_label: # read label from an atlas
            roi = mne.read_labels_from_annot(fs_id, parc=parc,
                                             subjects_dir=paths['fs'], 
                                             regexp=roi_label)[0]
        elif not parc and roi_label: # or morph a fsaverage label to individual
            roi = mne.read_label('%s/rois/%s.label' % (paths['cluster'], roi_label))
            roi = roi.copy().morph(subject_to=fs_id, subject_from='fsaverageJA', 
                                       subjects_dir=paths['fs'])
        else:
            roi = None
        
        _,itpc = mne.minimum_norm.source_induced_power(epochs, inv, freqs, 
                                                       roi, lambda2, 
                                                       method=stc_method, 
                                                       n_cycles=n_cycles, 
                                                       pick_ori=pick_ori,
                                                       decim=decim, 
                                                       n_jobs=num_jobs)
        if not vertexwise: # average over sources
            itpc = itpc.mean(0)
            
    itpcs.update({itpc_labels[i_epochs]: itpc[:,:,i_tmin:i_tmax]})
    
#         if vertexwise:
#             # average over times
#             itpc_stims[i_epochs] = itpc[:,:,i_tmin:i_tmax].mean(-1) 
#         else:
#             # average over sources and times 
#             itpc_stims[i_epochs, i_roi] = itpc[:,:,i_tmin:i_tmax].mean((0,-1))     

# # average itpc within conditions
# itpc_within = np.zeros((n_conds, n_spats, n_freqs))
# for i_cond,cond in enumerate(conds):
#     inds = [i for i,itpc_label in enumerate(itpc_labels) if 'within' in itpc_label 
#             and cond in itpc_label]
#     itpc_within[i_cond] = itpc_stims[inds].mean(0)
 
# inds = [i for i,itpc_label in enumerate(itpc_labels) if 'across' in itpc_label]
# itpc_across = itpc_stims[inds].mean(0)
        
# # calculate phase dissimilarity function for each condition
# print('\nCalculating phase dissimilarity functions (PDFs)', flush=True)
# pdfs = np.zeros((n_conds, n_spats, n_freqs))
# for i_cond,cond in enumerate(conds):
#     pdfs[i_cond] = itpc_within[i_cond]-itpc_across
    
# # determine peak coherence and frequency in the PDF
# max_coh = np.max(pdfs, axis=-1)
# max_freq = freqs[np.argmax(pdfs, axis=-1)]
           
data_struct = {'itpc': itpcs, 'conds': conds, 'rois': roi_labels, 
               'times': times, 'freqs': freqs, 'n_cycles': n_cycles, 
               'method': stc_method, 'lambda2': lambda2, 'pick_ori': pick_ori}

#% save ITPC/PDF data
save_path = '%s/%s/%s.p' % (paths['cluster'], sub_ID, fn_out)
print('Saving %s' % save_path)
f = open(save_path, 'wb')
pickle.dump(data_struct, f)
f.close()

# make stcs
# for i_cond,cond in enumerate(conds):
#     print('Saving PDFs for %s' % cond)
#     stc = mne.SourceEstimate(max_coh[i_cond], vertices=verts, tmin=0,
#                               tstep=tstep, subject=fs_id)
#     stc.save('%s/pdf_peak_coh_%s' % (sub_path, cond))
#     stc = mne.SourceEstimate(max_freq[i_cond], vertices=verts, tmin=0, 
#                               tstep=tstep, subject=fs_id)
#     stc.save('%s/pdf_peak_freq_%s' % (sub_path, cond))
    
print('All done!')
    
        
