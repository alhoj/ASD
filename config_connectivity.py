#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:21:32 2022

@author: ja151
"""

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'local': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

## set parameters
exclude = ['105801', '107301'] # exclude subjects

conds = ['MSS', 'SWS'] # conditions to include
equalize_epoch_counts = True # equalize epoch counts between conditions
add_baseline_cond = True

rois = ['AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh', # regions of interest
        'IP_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
        'IF_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh']

rois_as_seed = False # ROIs as seed or between ROIs
tmin = None # in s; None for beginning/ of epochs
tmax = None # in s; None for end of epochs
average_times = False # average times between tmin and tmax

fmin = 8
fmax = 14
fstep = 0.5
average_freqs = False # average frequencies between fmin and fmax

con_method = ['coh', 'imcoh', 'wpli'] # connectivity method to use; can be several
con_mode = 'cwt_morlet' # spectrum estimation mode
n_cycles = ('diff', 2) # e.g. ('fixed', 7) -> fixed 7 cycles; ('diff', 2) -> freqs / 2 cycles

stc_method = 'MNE' # source estimation method
lambda2 = 1.0 # regularization parameter for the source estimates
pick_ori = 'normal' # orientation of the source dipoles
tc_extract = 'mean_flip' # time course extraction method

n_jobs = 12 # how many epochs to process in parallel
overwrite = True

