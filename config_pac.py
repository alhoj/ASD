#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:52:27 2022

@author: ja151
"""

paths = {'cluster': '/autofs/cluster/transcend/jussi/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

## set variables

# exclude subjects
exclude = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']

# which conditions to calculate PAC for
conds = ['MSS', 'SWS'] # 'Speech/B', 'Jabber/B', 'Noise/B', 'MSS', 'SWS'

# time limits with respect to the stimulus onset (in seconds)
tmin = 0.
tmax = 1.5

# frequencies (Hz) for carrier (phase) and modulated (amp) signals
phase_fq_min = 8 # if 'PAF', use individual peak alpha frequency
phase_fq_max = 12
phase_fq_step = 0.5
phase_fq_width = 2
amp_fq_min = 30
amp_fq_max = 60
amp_fq_step = 3

# which method to use to calculate PAC
pac_method = 'ozkurt' # 'canolty', 'ozkurt', or 'duprelatour'
filter_method = 'pactools' # 'pactools' or 'mne' (only in 'canolty' and 'ozkurt')
concatenate_epochs = True
add_baseline = True # add baseline condition
baseline_length = 0.5
equalize_epoch_counts = 'cond' # if 'stim', equalize across stimuli
                               # if 'cond', equalize across conditions

# brain regions
roi_labels = ['AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh'
              ]
vertexwise = False # True -> use rois as mask and calculate PAC for each vertex within the mask
               # False -> average time courses within each ROI

# params for source estimates
snr = 1.0
lambda2 = 1.0 / snr ** 2
stc_method = 'MNE'
pick_ori = 'normal'
tc_extract = 'mean_flip' # only if vertexwise = False

# overwrite if out_ID file already exists
overwrite = True

# for the output file
postfix = None

# number of cores (for parallel processing)
n_jobs = 1
