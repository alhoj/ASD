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
import matplotlib.pyplot as plt
import os
import subprocess

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

f = open('%s/p/subjects.p' % paths['out'], 'rb')
sub_info = pickle.load(f)
f.close()

# define and remove bad subjects
# bad_subs = ['105801', '107301']
bad_subs = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for bad_sub in bad_subs:
    index = sub_info['sub_ID'].index(bad_sub)  
    [sub_info[key].pop(index) for key in sub_info.keys()]

n_ASD = len([i for i in sub_info['ASD'] if i=='Yes'])
n_TD = len([i for i in sub_info['ASD'] if i=='No'])

conds = ['MSS', 'SWS'] # 'MSS', 'SWS'], 'Speech', 'Jabber', 'Noise']

# Parameters for source estimate
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'
spacing = 'oct6'

tmin = 0.
tmax = 0.5
n_verts = 10
peak = True # peak or mean
equalize_epoch_counts = True
# mask within which fROI is defined; can be more than one
mask_labels = ['S_postcentral', 'S_intrapariet']#'seed_AC_peak0-200ms_wpli2_debiased_8-14Hz_28TD_vs_28ASD_MSS-SWS_cdt0.05_phtNone']#'supramarginal_div1', 'supramarginal_div2']#'parstriangularis', 'parsorbitalis', 'parsopercularis']  #,'inferiorparietal', , 
exclude_labels = []#'supramarginal', 'superiorparietal']#] # exclude these from mask
hemis = ['lh']#,'rh']
adjacent = True # in case of separate clusters, select the one with the peak value
prefix = 'S_postcentral_intraparietal'
postfix = 'auto'

#%% Read data

sub_IDs = sub_info['sub_ID']
for sub_ID in sub_IDs:                
    sub_path = '%s/%s' % (paths['cluster'], sub_ID)    
    epochs = mne.read_epochs('%s/%s_speech_1-30Hz_-200-2000ms-epo.fif' 
                             % (sub_path, sub_ID), proj=False)
    picks = mne.pick_types(epochs.info, meg=True)
    sfreq = epochs.info['sfreq']
    inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-30Hz-oct6-inv.fif' 
                                                 % (sub_path, sub_ID))
    src = inv['src']
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    verts = [s['vertno'] for s in src]

    if equalize_epoch_counts and len(conds)>1:
        epochs.equalize_event_counts(conds)
    erfs = epochs[conds].apply_baseline().average(picks=picks)
    stc = mne.minimum_norm.apply_inverse(erfs, inv, lambda2=lambda2, 
                                         method=method, verbose=True)
    stc.data = abs(stc.data)
    
    for hemi in hemis:
        # label = mne.read_labels_from_annot('fsaverageJA', parc='PALS_B12_Lobes', 
        #                                     hemi=hemi, subjects_dir=paths['fs'], 
        #                                     regexp='LOBE.TEMPORAL')[0]
        mask = []
        for roi in mask_labels:
            path_roi = '%s/rois/%s-%s.label' % (sub_path, roi, hemi)
            path_roi_fsave = '%s/rois/%s-%s.label' % (paths['cluster'], roi, hemi)
            if os.path.exists(path_roi):
                mask.append(mne.read_label(path_roi, subject=fs_id))
            elif os.path.exists(path_roi_fsave):
                roi = mne.read_label(path_roi_fsave, subject='fsaverageJA')
                mask.append(roi.copy().morph(subject_to=fs_id, 
                                             subject_from='fsaverageJA',
                                             grade=verts,
                                             subjects_dir=paths['fs']))
            else:
                try:
                    mask.append(mne.read_labels_from_annot(fs_id, parc='aparc', hemi=hemi, 
                                                             subjects_dir=paths['fs'], 
                                                             regexp=roi)[0])
                except:
                    mask.append(mne.read_labels_from_annot(fs_id, parc='aparc.a2009s', 
                                                             hemi=hemi, subjects_dir=paths['fs'], 
                                                             regexp=roi)[0])
        if len(mask)>1:
            temp = mask[0]
            for i in np.arange(1,len(mask)):
                temp += mask[i]
            mask = temp
        else:
            mask = mask[0]
            
            
        ##   
        if exclude_labels:
            exclude = []
            for roi in exclude_labels:
                path_roi = '%s/rois/%s-%s.label' % (sub_path, roi, hemi)
                path_roi_fsave = '%s/rois/%s-%s.label' % (paths['cluster'], roi, hemi)
                if os.path.exists(path_roi):
                    exclude.append(mne.read_label(path_roi, subject=fs_id))
                elif os.path.exists(path_roi_fsave):
                    roi = mne.read_label(path_roi_fsave, subject='fsaverageJA')
                    exclude.append(roi.copy().morph(subject_to=fs_id, 
                                                 subject_from='fsaverageJA',
                                                 grade=verts,
                                                 subjects_dir=paths['fs']))
                else:
                    try:
                        exclude.append(mne.read_labels_from_annot(fs_id, parc='aparc', hemi=hemi, 
                                                                  subjects_dir=paths['fs'], 
                                                                  regexp=roi)[0])
                    except:
                        exclude.append(mne.read_labels_from_annot(fs_id, parc='aparc.a2009s', 
                                                                  hemi=hemi, subjects_dir=paths['fs'], 
                                                                  regexp=roi)[0])
            if len(exclude)>1:
                temp = exclude[0]
                for i in np.arange(1, len(exclude)):
                    temp += exclude[i]
                exclude = temp
            else:
                exclude = exclude[0]
        ##

        if peak:
            _,i_time = stc.in_label(mask).get_peak(hemi=hemi, tmin=tmin, 
                                                   tmax=tmax, time_as_index=True)
        else:
            stc = stc.crop(tmin,tmax).mean()
            i_time = 0
        
        data = np.zeros(stc.data.shape[0])
        if hemi=='lh':
            inds_verts_mask = mask.get_vertices_used(verts[0])
            if exclude_labels:
                inds_verts_exclude = exclude.get_vertices_used(verts[0])
                inds_verts_mask = list(set(inds_verts_mask)-set(inds_verts_exclude))            
            inds_verts = np.searchsorted(verts[0], inds_verts_mask)
            data[inds_verts] = stc.copy().data[inds_verts, i_time]
            data[len(verts[0])::] = 0
        else:
            inds_verts_mask = mask.get_vertices_used(verts[1])
            if exclude_labels:
                inds_verts_exclude = exclude.get_vertices_used(verts[1])
                inds_verts_mask = list(set(inds_verts_mask)-set(inds_verts_exclude))  
            inds_verts = len(verts[0]) + \
                np.searchsorted(verts[1], mask.get_vertices_used(verts[1]))
            data[inds_verts] = stc.copy().data[inds_verts, i_time]
            data[0:len(verts[0])] = 0
            
            
        # data = abs(data)
        # threshold of highest n_verts values
        th = np.sort(np.squeeze(data))[::-1][n_verts]
        # indices of top n_verts vertices
        inds = data>th
        
        # make stc with the thresholded data
        data_th = np.zeros(data.shape)
        data_th[inds] = data[inds]
        stc_th = mne.SourceEstimate(data_th, verts, tmin, tstep=0, 
                                    subject=fs_id)
        
        
        # make ROI from stc
        if adjacent:
            froi = mne.stc_to_label(stc_th, src=src, smooth=True, connected=True,
                                    subjects_dir=paths['fs'])[0]
            froi = froi[0]
        else:
            froi = mne.stc_to_label(stc_th, src=src, smooth=True,
                                   subjects_dir=paths['fs'])
            froi = list(filter(None, froi))[0]
            
            
        
        if peak:
            froi_name = '%s_%s_%dverts_%s_peak%d-%dms' % (prefix, method, n_verts, ('_').join(conds),
                                                            int(tmin*1000), int(tmax*1000))
        else:
            froi_name = '%s_%s_%dverts_%s_mean%d-%dms' % (prefix, method, n_verts, ('_').join(conds),
                                                            int(tmin*1000), int(tmax*1000))
        if adjacent:
            froi_name += '_adjVerts'
            
        froi_name += '_%s' % postfix
            
        # save ROI png
        out_folder = '%s/figures/misc/rois/%s/' % (paths['cluster'], froi_name)
        if not os.path.exists(out_folder):
            subprocess.call('mkdir %s' % out_folder, shell=True)
        brain = stc_th.plot(subject=fs_id, subjects_dir=paths['fs'], hemi=hemi, 
                            backend='matplotlib', spacing=spacing)
        brain.savefig('%s/%s_%s-%s.png' % (out_folder, sub_ID, froi_name, hemi), dpi=300)
        plt.close()
        # save ROI
        froi.save('%s/rois/%s' % (sub_path, froi_name))
        
        # visualize ROI
        # data_roi = np.zeros(data.shape)
        # stc = mne.SourceEstimate(data_roi, verts, tmin, tstep=0.001, subject=fs_id)
        # if hemi=='lh':
        #     inds_verts = np.searchsorted(verts[0], roi.get_vertices_used(verts[0]))
        #     data_roi[inds_verts] = 1
        # else:
        #     inds_verts = len(verts[0]) + np.searchsorted(verts[1], 
        #                                                  roi.get_vertices_used(verts[1]))
        #     data_roi[inds_verts] = 1
        # stc = mne.SourceEstimate(data_roi, verts, tmin, tstep=0.001, subject=fs_id)
        # stc.plot(subject=fs_id, subjects_dir=paths['fs'], hemi=hemi, 
        #          backend='matplotlib', spacing='oct6')
    
    
    
