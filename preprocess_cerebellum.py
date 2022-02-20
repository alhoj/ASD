#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:48:33 2021

@author: ja151
"""

import pickle
import mne
import os

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'local': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/',
         'cb': '/autofs/cluster/transcend/data_exchange/cerebellar_source_spaces_autism_cohort/source_spaces'
         }


f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# define and remove bad subjects
bad_subs = ['105801', '107301', '100001']
for bad_sub in bad_subs:
    ind = sub_info['sub_ID'].index(bad_sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
     if len(sub_info[key])==n_subs]
sub_IDs = sub_info['sub_ID']

#%%
for sub_ID in sub_IDs:
    print(sub_ID)
    
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    fn_src = '%s/%s-src.fif' % (paths['cb'], fs_id)   
    fn_trans = '%s/%s//%s_speech-trans.fif' % (paths['cluster'], sub_ID, sub_ID)
    fn_bem = '%s/%s/%s-5120-bem-sol.fif' % (paths['cluster'], sub_ID, sub_ID)
    fn_fwd = '%s/%s/%s_speech-cb-fwd.fif' % (paths['local'], sub_ID, sub_ID)
    fn_cov = '%s/%s/%s_erm_1-30Hz-cov.fif' % (paths['cluster'], sub_ID, sub_ID)
    fn_inv = '%s/%s/%s_erm_1-30Hz-cb-inv.fif' % (paths['local'], sub_ID, sub_ID)
    
    info = mne.io.read_info('%s/%s/%s_speech_1-30Hz_-200-2000ms-epo.fif' 
                            % (paths['cluster'], sub_ID, sub_ID), verbose='warning')
    
    if not os.path.exists(fn_bem):
        model = mne.make_bem_model(subject=fs_id, ico=4, subjects_dir=paths['fs'], 
                                   conductivity=[0.3])
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(fn_bem, bem)
            
    if not os.path.exists(fn_fwd):
        fwd = mne.make_forward_solution(info, fn_trans, fn_src, fn_bem)
        mne.write_forward_solution(fn_fwd, fwd)
    else:
        fwd = mne.read_forward_solution(fn_fwd)
        
    cov = mne.read_cov(fn_cov)
    
    if not os.path.exists(fn_inv):
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose='auto', 
                                                     depth=0.8,rank='info')
        mne.minimum_norm.write_inverse_operator(fn_inv, inv)
    