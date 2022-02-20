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
import pickle
import numpy as np
import os

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'local': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# define bad subjects
# bad_subs = ['105801', '107301']
bad_subs = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for bad_sub in bad_subs:
    ind = sub_info['sub_ID'].index(bad_sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
      if len(sub_info[key])==n_subs]
sub_IDs = sub_info['sub_ID']
# sub_IDs = ['083701', '089201', '093101']


# conditions
conds = ['all']
n_conds = len(conds)
equalize_event_counts = False
combine_conds = False

# rois
roi_names = ['AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
             'seed_AC_wpli2_debiased_8-12Hz_28TD_vs_28ASD_MSS-SWS_cdt0.01_pht1e-08-lh'
             ]
roi_labels = ['AC', 'con_phte-8']
average_verts = True # False, True, or ROI-specific vector

save_prefix = 'ACseedConClusters'

# params for source estimates
snr = 1.0
lambda2 = 1.0 / snr ** 2
stc_method = 'MNE'
pick_ori = 'normal'
tc_extract = 'mean_flip'

# Read the source space 
src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
src_fsave = mne.read_source_spaces(src_fname)
verts_fsave = [s['vertno'] for s in src_fsave]

#%% 

for sub_ID in sub_IDs:
    local_path = '%s/%s/' % (paths['local'], sub_ID)
    cluster_path = '%s/%s/' % (paths['cluster'], sub_ID)
    inv = mne.minimum_norm.read_inverse_operator('%s/%s_0-200Hz-oct6-inv.fif' 
                                                 % (local_path, sub_ID))
    src = inv['src']    
    verts = [s['vertno'] for s in src]
    n_verts = len(np.concatenate(verts))
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    
    epochs = mne.read_epochs('%s/%s_speech_0-200Hz_notch60Hz_-500-2000ms-epo.fif' 
                             % (local_path, sub_ID), proj=False)
    sfreq = epochs.info['sfreq']
    
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
            
    if combine_conds:
        this_conds = [conds]
    else:
        this_conds = conds
    
    rois = []
    for i,roi_name in enumerate(roi_names):
        roi_path = '%s/%s/rois/%s.label' % (paths['cluster'], sub_ID, 
                                            roi_name)
        roi_path_fsave = '%s/rois/%s.label' % (paths['cluster'], roi_name)
        if os.path.exists(roi_path):
            rois.append(mne.read_label(roi_path, subject=fs_id))
        elif os.path.exists(roi_path_fsave):
            if type(average_verts)==list:
                this_average_verts = average_verts[i]
            else:
                this_average_verts = average_verts
            if average_verts:
                roi = mne.read_label(roi_path_fsave, subject='fsaverageJA')
                rois.append(roi.copy().morph(subject_to=fs_id, 
                                             subject_from='fsaverageJA',
                                             grade=verts,
                                             subjects_dir=paths['fs']))
            else:
                rois.append(mne.read_label(roi_path_fsave, subject='fsaverageJA'))

                
        else:
            rois.append(mne.read_labels_from_annot('fsaverageJA', 
                                                    parc='PALS_B12_Lobes', 
                                                    subjects_dir=paths['fs'], 
                                                    regexp=roi_name)[0])
    
    
    for cond in this_conds:
        print('\nCondition: %s' % cond)
        if average_verts or all(average_verts):
            if cond=='all':
                this_epochs = epochs
            else:
                this_epochs = epochs[cond]
            stcs = mne.minimum_norm.apply_inverse_epochs(this_epochs, 
                                                         inverse_operator=inv, 
                                                         lambda2=lambda2, 
                                                         method=stc_method, 
                                                         pick_ori=pick_ori) 
            tcs = mne.extract_label_time_course(stcs, rois, src, tc_extract)
            info = mne.create_info(roi_labels, sfreq)
        else:
            tcs = []
            labels = []
            for i,roi in enumerate(rois):
                if cond=='all':
                    this_epochs = epochs
                else:
                    this_epochs = epochs[cond]
                stcs = mne.minimum_norm.apply_inverse_epochs(this_epochs, 
                                                             inverse_operator=inv, 
                                                             lambda2=lambda2, 
                                                             method=stc_method, 
                                                             pick_ori=pick_ori)
                if average_verts[i]:
                    tcs.append(mne.extract_label_time_course(stcs, roi, src, tc_extract))
                    labels.append([roi.name[0:2]])
                else:
                    # morph individual source estimate to freesurfer average brain
                    morph = mne.compute_source_morph(src, fs_id, 'fsaverageJA',
                                                     src_to=src_fsave, 
                                                     subjects_dir=paths['fs'])        
                    stcs = [morph.apply(stc) for stc in stcs]
                    tcs.append([stc.in_label(roi).data for stc in stcs])
                    labels.append([str(i) for i in range(tcs[i][0].shape[0])])                    
                    
            tcs = np.concatenate(tcs, axis=1)
            info = mne.create_info(list(np.concatenate(labels)), sfreq)
            
        
        # put time courses in mne data array  
        data = mne.EpochsArray(tcs, info, tmin=-0.5, baseline=(-0.5, 0))
        if combine_conds:
            cond = ('_').join(cond)
        data.save('%s/%s_%s_%d-%dms_%d-%dHz-epo.fif' % (cluster_path, save_prefix, cond,
                                                        int(this_epochs.tmin * 1000),
                                                        int(this_epochs.tmax * 1000),
                                                        int(this_epochs.info['highpass']),
                                                        int(this_epochs.info['lowpass'])), overwrite=True)
