#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:22:28 2021

@author: ja151
"""

import pickle
import subprocess

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'cluster': '/autofs/cluster/transcend/jussi/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

## read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

## exclude subjects
# exclude = ['105801', '107301']
exclude = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for sub_ID in exclude:
    ind = sub_info['sub_ID'].index(sub_ID)
    sub_info['sub_ID'].pop(ind)

## function command in the slurm jobs
func = 'python3 export_ROI_timecourses_slurm.py' # 'source make_watershed_bem.csh'

## other slurm parameters
partition = 'basic' # 'basic', 'rtx6000', 'rtx8000'
mem = '80G'
time = '00:30:00'
gpus = 6 # only if partition rtx6000 or rtx8000

## empty jobs and logs folders
subprocess.call('rm %s/jobs/*' % paths['cluster'], shell=True)
subprocess.call('rm %s/logs/*' % paths['cluster'], shell=True)

sub_IDs = sub_info['sub_ID']
# sub_IDs = ['010401']

## write jobs (one per subject)
for i, sub_ID in enumerate(sub_IDs[1::]):
    # i += 100
    print(sub_ID)
    job_name = '%s/jobs/job%d.sh' % (paths['cluster'], int(i+1))
    log_name = '%s/logs/log%d.txt' % (paths['cluster'], int(i+1))
    f = open(job_name, 'x')
    ## write basic slurm parameters
    f.writelines(['#!/bin/bash\n',
                  '#SBATCH --account=transcend\n',
                  '#SBATCH --partition=%s\n' % partition,
                  '#SBATCH --mem=%s\n' % mem,
                  '#SBATCH --time=0-%s\n' % time,
                  '#SBATCH --output=%s\n' % log_name])
    ## if other than basic partition is used, include number of gpus
    if partition!='basic': f.writelines(['#SBATCH --gpus=%d\n' % gpus])
    ## write the script execution commands
    f.writelines(['\ncd %s/scripts/\n' % paths['cluster'],
                  '%s %s' % (func, sub_ID)])
    f.close()
    