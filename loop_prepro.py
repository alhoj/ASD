#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:25:21 2020

@author: ja151
"""
import sys
sys.path.append('/local_mount/space/hypatia/1/users/jussi/scripts/')
from MNEprepro import MNEprepro
import pickle

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

sub_info = pickle.load(open("/local_mount/space/hypatia/1/users/jussi/speech/subjects.p", "rb"))

# define and remove bad subjects
bad_subs = ['011202', '052402', '082802', '097601', '099302']
for bad_sub in bad_subs:
    index = sub_info['sub_ID'].index(bad_sub)
    sub_info['sub_ID'].pop(index)
    sub_info['FS_dir'].pop(index)
    sub_info['ASD'].pop(index)

for sub_ID in sub_info['sub_ID']:
    if sub_ID.isnumeric():      
        sub = MNEprepro(sub_ID, paths)
        sub.detect_bad_channels()
        sub.run_maxwell_filter(tSSS=True)
        sub.run_maxwell_filter()
        sub.run_ICA()
        sub.get_epochs(tmax=2.0, reject=None, postfix='_0-40Hz_-200-2000ms_noReject')
#        sub.get_epochs(tmax=2.0, reject=dict(grad=3000e-13, mag=5e-12), postfix='_0-40Hz_-200-2000ms_rejectGrad3Mag5')
#        sub.forward_modeling()
#        sub.compute_noise_cov(erm=True, method='shrunk', rank={'meg': 72}, overwrite=True)
#        sub.compute_noise_cov(erm=False, method='empirical', rank={'meg': 72})
#        sub.make_inv_operator(postfix='-shrunk-reducedRank', overwrite=True)