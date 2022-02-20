#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:00:03 2021

@author: ja151
"""

import mne
import numpy as np
import pickle
import scipy
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, ranksums, pearsonr
from matplotlib import pyplot as plt
import sys
import subprocess
import seaborn as sns
import pandas as pd
import pingouin as pg
from helper_functions import compare_corr_coefs

sys.path.append('/autofs/cluster/transcend/jussi/tools/')

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'cluster': '/autofs/cluster/transcend/jussi/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# bad_subs = ['105801', '107301']
## equalized NVIQs - new sample N_ATD=29, N_ASD=29
# bad_subs = ['105801', '107301', '052902', '090902', '048102']
## equalized NVIQs - new sample N_ATD=28, N_ASD=28
bad_subs = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for bad_sub in bad_subs:
    ind = sub_info['sub_ID'].index(bad_sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
     if len(sub_info[key])==n_subs]
sub_IDs = sub_info['sub_ID']
n_subs = len(sub_IDs)
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
n_ASD = len(inds_ASD)
n_TD = len(inds_TD)

groups = ['TD', 'ASD']
covar_labels = [] # 'age', 'NVIQ', 'VIQ'

fn_in = 'pac_9ROIs_MSS-SWS-Baseline_8-12Hz_30-100Hz_0-1500ms_ozkurt_concatEpochs_ACseedConCluster.p'

#% load one file to get variables
f = open('%s/%s/p/%s' % (paths['cluster'], sub_info['sub_ID'][0], fn_in), 'rb')
data = pickle.load(f)
f.close()

conds = list(data['pac'].keys())
conds = [cond.replace('/','') for cond in conds]
n_conds = len(conds)

rois = data['rois']
n_rois = len(rois)

pac_method = data['pac_method']

[tmin, tmax] = data['times']

freqs_phase = data['phase_fq_range'] # data['fphase_range']
phase_fq_min = freqs_phase[0]
phase_fq_max = freqs_phase[-1]
if phase_fq_min != phase_fq_max:
    phase_fq_step = np.diff(freqs_phase)[0]
n_phase_fq = len(freqs_phase)

freqs_amp = data['amp_fq_range'] # data['famp_range']
amp_fq_min = freqs_amp[0]
amp_fq_max = freqs_amp[-1]
if amp_fq_min != amp_fq_max:
    amp_fq_step = np.diff(freqs_amp)[0]
n_amp_fq = len(freqs_amp)

pac_method = data['pac_method']
concat_epochs = data['concatenate_epochs']


#%% load data

contrast = 'subtract' # 'subtract', 'parametric', or 'nonparametric'
                        # either subtract PAC values between conditions or 
                        # use parametric paired-samples t-test or
                        # nonparametric wilcoxon signed-rank test 
group_cond = np.zeros((n_conds, n_subs, n_rois, n_phase_fq, n_amp_fq))
n_contrasts = int(scipy.special.comb(n_conds, 2))
group_norm = np.zeros((n_contrasts, n_subs, n_rois, n_phase_fq, n_amp_fq))

for i_sub,sub_ID in enumerate(sub_info['sub_ID']):
    # print('\nSubject: %s' % sub_ID)
    # subject path
    sub_path = '%s/%s/p/' % (paths['cluster'], sub_ID)
    try:
        f = open('%s/%s' % (sub_path, fn_in), 'rb')
        data = pickle.load(f)
        f.close()
    except:
        print('Could not find %s for %s' % (fn_in, sub_ID))

    cond_data = []
    n_epochs = data['n_epochs']
    for i_cond,cond in enumerate(conds):
        if data['concatenate_epochs']:
            cond_data.append(np.asarray(data['pac'][cond]))
            group_cond[i_cond, i_sub] = cond_data[i_cond].squeeze()
        else:
            # take median across epochs and store in array
            group_cond[i_cond, i_sub] = np.median(np.asarray(data['pac'][cond]), axis=-1)
            # reshape 
            cond_data.append(np.asarray(data['pac'][cond]).reshape(
                n_rois*n_phase_fq*n_amp_fq, n_epochs))
        
    i_contrast = 0
    stats = np.zeros((n_rois*n_phase_fq*n_amp_fq))
    contrast_labels = []
    for i_cond in np.arange(0,n_conds-1):
        for j_cond in np.arange(i_cond+1,n_conds):
            contrast_labels.append('%s-%s' % (conds[i_cond].replace('/',''), 
                                          conds[j_cond].replace('/','')))
            if data['concatenate_epochs']:
                group_norm[i_contrast, i_sub] = cond_data[i_cond].squeeze() - \
                                            cond_data[j_cond].squeeze()
            else:
                if contrast=='subtract':
                    print('Subtracting %s from %s' % (conds[j_cond], conds[i_cond]))
                    group_norm[i_contrast, i_sub] = \
                        np.median(cond_data[i_cond], axis=-1).reshape(n_rois,n_phase_fq,n_amp_fq) - \
                        np.median(cond_data[j_cond], axis=-1).reshape(n_rois,n_phase_fq,n_amp_fq)
                elif contrast=='nonparametric':
                    print('Calculating Wilcoxon signed=rank test %s vs. %s' 
                          % (conds[i_cond], conds[j_cond]))
                    for i in range(n_rois*n_phase_fq*n_amp_fq):
                        stats[i],_ = wilcoxon(cond_data[i_cond][i], 
                                              cond_data[j_cond][i],
                                              alternative='greater')
                    group_norm[i_contrast, i_sub] = stats.reshape(n_rois, n_phase_fq, n_amp_fq)
                elif contrast=='parametric':
                    print('Calculating paired-samples t-test %s vs. %s' 
                          % (conds[i_cond], conds[j_cond]))
                    for i in range(n_rois*n_phase_fq*n_amp_fq):
                        stats[i],_ = ttest_rel(cond_data[i_cond][i], cond_data[j_cond][i])
                    group_norm[i_contrast, i_sub] = stats.reshape(n_rois, n_phase_fq, n_amp_fq)
                        
            i_contrast += 1

data_orig = list(np.concatenate((group_cond,group_norm)))

conds = conds + contrast_labels
n_conds = len(conds)

#%% Load data alt

data = np.zeros((n_subs, n_conds, n_rois, n_phase_fq, n_amp_fq))

for i_sub,sub_ID in enumerate(sub_info['sub_ID']):
    # subject path
    sub_path = '%s/%s/p/' % (paths['cluster'], sub_ID)
    f = open('%s/%s' % (sub_path, fn_in), 'rb')
    data_sub = pickle.load(f)
    f.close()
        
    for i_cond,cond in enumerate(conds):
        data[i_sub, i_cond] = np.asarray(data_sub['pac'][cond]).squeeze()

#% do contrasts
data_contrasts = []
contrast_labels = []
for i_cond,cond1 in enumerate(conds[0:-1]):
    for j_cond,cond2 in enumerate(conds[i_cond+1::], i_cond+1):
        data_contrasts.append(data[:, i_cond] - data[:, j_cond])
        contrast_labels.append('%s-%s' % (cond1, cond2))
data_contrasts = np.array(data_contrasts).transpose(1,0,2,3,4)

# add contrasts data
data = np.concatenate((data, data_contrasts), axis=1)
conds = conds + contrast_labels
n_conds = len(conds)
        
data_orig = data
#% Regress covariates    
if covar_labels:
    print('Regressing out covariates: %s' % covar_labels)
    covar_out = '%s' % ('').join(covar_labels)
    covars = []
    for covar_label in covar_labels:
        covars.append(sub_info[covar_label])
    model = np.zeros((n_subs,len(covars)+1))
    model[:,0] = np.ones(n_subs)
    model[:,1::] = np.transpose(covars)
       
    # data = np.transpose(data_orig, (1,0,2,3,4))
    data = np.reshape(data, (n_subs, n_conds*n_rois*n_phase_fq*n_amp_fq))
    beta = scipy.linalg.lstsq(model, data)[0]
    data = data - model.dot(beta)
    # intercept = beta[0]
    # data += intercept
    data = np.reshape(data, (n_subs, n_conds, n_rois, n_phase_fq, n_amp_fq))
    # data = np.transpose(data, (1,0,2,3,4))

else: 
    # data = data_orig
    print('No covariates')
    covar_out = 'no'

#% put data in dict
    
data_dict = {}
for cond, data in zip(conds, list(np.transpose(data, (1,0,2,3,4)))):
    data_dict.update({cond: data})
    
data_orig_dict = {}
for cond, data in zip(conds, list(np.transpose(data_orig, (1,0,2,3,4)))):
    data_orig_dict.update({cond: data})
    
#%% plot & report
    
cois = ['MSS-SWS'] # conditions of interest
roi_id = 'cdt0.001_phtNone_MNE_5verts_MSS_SWS_mean0-1500ms_adjVerts' 
hemi = 'lh'

#freq limits for plotting
fmin_phase = 8 # Hz
fmax_phase = 14
fmin_amp = 30 
fmax_amp = 100

cdt_p = 0.05 # cluster-defining threshold

zscore = False

fontsize = 5

inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
TD_IDs = [sub_IDs[i] for i in inds_TD]
ASD_IDs = [sub_IDs[i] for i in inds_ASD]
TD_ages = [sub_info['age'][i] for i in inds_TD]
ASD_ages = [sub_info['age'][i] for i in inds_ASD]
inds_TD_ageSorted = np.argsort(TD_ages)
inds_ASD_ageSorted = np.argsort(ASD_ages)
TD_IDs_ageSorted = [TD_IDs[i] for i in inds_TD_ageSorted]
ASD_IDs_ageSorted = [ASD_IDs[i] for i in inds_ASD_ageSorted]
sub_IDs_sorted = TD_IDs_ageSorted + ASD_IDs_ageSorted

i_fmin_phase = list(freqs_phase).index(fmin_phase)
i_fmax_phase = list(freqs_phase).index(fmax_phase)
i_fmin_amp  = list(freqs_amp).index(fmin_amp)
i_fmax_amp  = list(freqs_amp).index(fmax_amp)
this_freqs_phase = freqs_phase[i_fmin_phase:i_fmax_phase+1]
this_freqs_amp = freqs_amp[i_fmin_amp:i_fmax_amp+1]
this_n_phase_fq = len(this_freqs_phase)
this_n_amp_fq = len(this_freqs_amp)

result_IDs = ['TD group mean', 'ASD group mean', 
            'TD vs. ASD t-test', 
            'TD vs. ASD t-test at p=.05',
            'TD vs. ASD permutation cluster test at p=.05']

score_labels = ['age', 'NVIQ', 'VIQ', 'ICSS', 'SCSS', 'ASPS', 'SRS_tot_T', 
                'SRS_socComm_T', 'ADOS_tot_new', 'ADOS_comm_soc_new'] # 'age', 'NVIQ', 'VIQ', 
result_IDs += ['%s_%s' % (group, score_label) for score_label in score_labels 
             for group in ['TD', 'ASD']]

for cond, data_cond in data_dict.items():
    if cond not in cois: continue
    data_cond = data_cond[:,:,i_fmin_phase:i_fmax_phase+1, i_fmin_amp:i_fmax_amp+1]
    
    print('Condition: %s' % cond)
    fn_save = '%s/html/pac_%dTD_%dASD_%s_phase%d-%dHz_amp%d-%dHz_%s' % (paths['cluster'], 
                                                                           n_TD, n_ASD, cond, 
                                                                           fmin_phase, fmax_phase, 
                                                                           fmin_amp, fmax_amp, 
                                                                           pac_method)
    if concat_epochs: fn_save += '_concatEpochs'
    fn_save += '_%s' % hemi
    fn_save += '_%sROIs' % roi_id
    fn_save += '_%sReg' % covar_out
    
    report = mne.report.Report()
    if zscore: data_cond = scipy.stats.zscore(data_cond, axis=None)
    
    figs = []
    captions = []
    for result_ID in sub_IDs_sorted + result_IDs:
        if result_ID.isdigit():            
            i_sub = sub_info['sub_ID'].index(result_ID)
            age = sub_info['age'][sub_info['sub_ID'].index(result_ID)]
            
            if sub_info['ASD'][sub_info['sub_ID'].index(result_ID)]=='Yes':
                group_label = 'ASD'
            else:
                group_label = 'TD'
            captions.append('%s - %.1f yo %s' % (result_ID, age, group_label))
        else:
            captions.append(result_ID)
            
        if hemi=='bihemi':
            fig, axes = plt.subplots(nrows=3, ncols=2, sharey='all', 
                                     figsize=(4,6), dpi=300)
            this_rois = rois
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', 
                                     figsize=(8,2), dpi=300)
            if hemi=='lh':
                this_rois = [roi for roi in rois if 'lh' in roi]
            else:
                this_rois = [roi for roi in rois if 'rh' in roi]
                
        this_rois = [roi for roi in this_rois if roi_id in roi]
            
        for roi in this_rois:
            i_roi = rois.index(roi)
            
            if roi.startswith('AC'):
                roi_label = 'AC'
                if hemi=='bihemi': 
                    i_row = 0
                else: 
                    i_col = 0
            elif roi.startswith('IP'):
                roi_label = 'IP'
                if hemi=='bihemi': 
                    i_row = 1
                else: 
                    i_col = 1
            elif roi.startswith('IF'):
                roi_label = 'IF'
                if hemi=='bihemi': 
                    i_row = 2
                else: 
                    i_col = 2
            
            if roi.endswith('lh'):
                roi_label += '-lh'
                if hemi=='bihemi': 
                    i_col = 0
            else:
                roi_label += '-rh'
                if hemi=='bihemi': 
                    i_col = 1
                
            if hemi=='bihemi':
                ax = axes[i_row,i_col]
            else:
                ax = axes[i_col]
                
            if result_ID.isdigit():
                data_plot = data_cond[i_sub,i_roi]
                vmin = np.min(data_cond[i_sub]) * 0.8
                vmax = np.max(data_cond[i_sub]) * 0.8
            elif result_ID=='TD group mean':
                data_plot = data_cond[inds_TD,i_roi].mean(0)
                # vmin = np.min(data_cond.mean(0)) * 0.8
                # vmax = np.max(data_cond.mean(0)) * 0.8 
                vmin = -0.003
                vmax = 0.003 
            elif result_ID=='ASD group mean':
                data_plot = data_cond[inds_ASD,i_roi].mean(0)
                # vmin = np.min(data_cond.mean(0)) * 0.8
                # vmax = np.max(data_cond.mean(0)) * 0.8 
                vmin = -0.003
                vmax = 0.003
            elif 't-test' in result_ID:
                t,p = scipy.stats.ttest_ind(data_cond[inds_TD,i_roi],
                                            data_cond[inds_ASD,i_roi])
                if 'at p=.05' in result_ID:
                    t_th = np.zeros(t.shape)
                    t_th[np.where(p<0.05)] = t[np.where(p<0.05)]
                    data_plot = t_th
                    vmin = -2
                    vmax = 2                    
                else:
                    data_plot = t
                    vmax = data_plot.max() * 0.8
                    vmin = data_plot.min() * 0.8
                    
            elif 'permutation cluster' in result_ID:
                # th = dict(start=0, step=0.2)
                th_p = cdt_p
                th = scipy.stats.distributions.f.ppf(1. - th_p / 2., n_ASD - 1, n_TD - 1)
                n_perms = 5000
                X = [data_cond[inds_TD,i_roi], data_cond[inds_ASD,i_roi]]
                stats, clusters, cluster_pvals, _ = mne.stats.permutation_cluster_test(
                    X, threshold=th, n_jobs=12, out_type='mask', n_permutations=n_perms+1,
                    seed=7)
                good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
                cluster_mask = np.zeros((n_phase_fq, n_amp_fq))
                if good_cluster_inds.size > 0:
                    print('Min cluster p-val: %.3f\n' % min(cluster_pvals))
                    stats_th = np.zeros(stats.shape)
                    if type(th)==dict:
                        stats_th[good_cluster_inds] = \
                            stats[good_cluster_inds]
                    else:
                        for good_cluster_ind in good_cluster_inds:
                            stats_th[clusters[good_cluster_ind]] = \
                                stats[clusters[good_cluster_ind]]
                    data_plot = stats_th
                    cluster_mask[np.nonzero(data_plot)] = 1
                    vmax = stats_th.max() * 0.8
                    vmin = -vmax
                else:
                    data_plot = np.zeros(stats.shape)
                    vmin = -1
                    vmax = 1
                    
            else: # correlations            
                brains = data_cond[:, i_roi].reshape(n_subs, this_n_phase_fq*this_n_amp_fq)
                scores = np.array(sub_info[('_').join(result_ID.split('_')[1::])])
                diagnosed = sub_info['ASD'].copy()
                
                vmin = -0.5
                vmax = 0.5
                
                # indices of None, i.e. missing scores
                null_inds = [i for i in range(len(scores)) 
                             if scores[i]==None]
                null_inds.reverse() # reverse so that largest index is removed first
                # remove Nones and the corresponding brain data
                for i in null_inds:
                    scores = np.delete(scores, i)
                    brains = np.delete(brains, i, axis=0)
                    diagnosed.pop(i)   
                    
                if 'TD' in result_ID:
                    inds_group = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
                else:
                    inds_group = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
                                   
                if not len(scores[inds_group])==0:
                    r = np.zeros((this_n_phase_fq*this_n_amp_fq))
                    p = np.zeros((this_n_phase_fq*this_n_amp_fq))
                    for i in np.arange(this_n_phase_fq*this_n_amp_fq):
                        r[i],p[i] = pearsonr(scores[inds_group], brains[inds_group, i])
                    r = r.reshape((this_n_phase_fq,this_n_amp_fq))
                    p = p.reshape((this_n_phase_fq,this_n_amp_fq))
                    
                    data_plot = r
                else:
                    data_plot = np.zeros((this_n_phase_fq, this_n_amp_fq))
                    
            im = ax.imshow(data_plot.T, 
                           extent=[fmin_phase, fmax_phase, fmin_amp, fmax_amp],
                           aspect='auto', origin='lower', cmap='RdBu_r', 
                           vmin=vmin, vmax=vmax, interpolation='gaussian')
            cbar = fig.colorbar(im, ax=ax) # , ticks=np.arange(vmin,vmax,2)
            cbar.ax.tick_params(labelsize=fontsize)
            ax.set_title('%s' % roi_label, fontsize=fontsize)
            # ax.set_xticks(np.arange(fmin_amp,fmax_amp,5))
            # ax.set_yticks(np.arange(fmin_phase,fmax_phase,2))
            ax.tick_params(axis='both', labelsize=fontsize)
    
        fig.text(.5, .02, 'Phase frequency (Hz)', ha='center', va='center', 
                 rotation='horizontal', fontsize=fontsize)
        fig.text(.02, .5, 'Amplitude frequency (Hz)', ha='center', va='center', 
                 rotation='vertical', fontsize=fontsize)
        plt.tight_layout()
            
        figs.append(fig)
        plt.close()
    
      
    report.add_figs_to_section(figs, captions=captions)
    report.save('%s.html' % fn_save, overwrite=True, open_browser=False)
    
    
  
#%% Post-hoc plot
    
roi = rois[7]
cond = 'MSS-SWS' 

# freq limits for plotting
fmin_phase = 8 # Hz
fmax_phase = 12
fmin_amp = 30 
fmax_amp = 60

# cluster-defining threshold
cdt_p = 0.05
pht_p = None # post-hoc threshold
n_perms = 1000
foi_phase = (8, 12)
foi_amp = (30, 60)
foi_mode = 'include' # 'include' or 'exclude' foi

zscore = False

i_fmin_phase = list(freqs_phase).index(fmin_phase)
i_fmax_phase = list(freqs_phase).index(fmax_phase)
i_fmin_amp  = list(freqs_amp).index(fmin_amp)
i_fmax_amp  = list(freqs_amp).index(fmax_amp)
this_freqs_phase = freqs_phase[i_fmin_phase:i_fmax_phase+1]
this_freqs_amp = freqs_amp[i_fmin_amp:i_fmax_amp+1]

inds_foi_phase = (list(freqs_phase).index(foi_phase[0]), 
                  list(freqs_phase).index(foi_phase[1]))
inds_foi_amp  = (list(freqs_amp).index(foi_amp[0]),
                 list(freqs_amp).index(foi_amp[1]))


i_roi = rois.index(roi)
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
        
data = data_dict[cond]
data_orig = data_orig_dict[cond]
if cdt_p=='TFCE':
    th = dict(start=0, step=1)
else:
    th = scipy.stats.distributions.f.ppf(1. - cdt_p  / 2., n_ASD - 1, n_TD - 1)

X = [data[inds_TD, i_roi], data[inds_ASD, i_roi]]
stats, clusters, cluster_pvals, _ = mne.stats.permutation_cluster_test(
    X, threshold=th, n_jobs=12, out_type='mask', n_permutations=n_perms+1,
    seed=7)
good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
cluster_mask = np.zeros((n_phase_fq, n_amp_fq), dtype=bool)
print('Min cluster p-val: %.3f\n' % min(cluster_pvals))
if good_cluster_inds.size > 0:
    if type(th)==dict:
        stats = stats.reshape(n_phase_fq*n_amp_fq)
        stats_th = np.zeros(stats.shape)
        stats_th[good_cluster_inds] = stats[good_cluster_inds]
        stats_th = stats_th.reshape((n_phase_fq,n_amp_fq))
    else:
        stats_th = np.zeros(stats.shape)
        for good_cluster_ind in good_cluster_inds:
            stats_th[clusters[good_cluster_ind]] = \
                stats[clusters[good_cluster_ind]]
                
    if pht_p:
        pht_F = scipy.stats.distributions.f.ppf(1. - pht_p / 2., 
                                                   n_ASD - 1, n_TD - 1)
        stats_th[np.where(stats_th < pht_F)] = 0
        
    data_plot = stats_th
    cluster_mask[np.nonzero(data_plot)] = True
    
    foi = np.zeros((n_phase_fq, n_amp_fq), dtype=bool)
    foi[inds_foi_phase[0]:inds_foi_phase[1]+1, 
        inds_foi_amp[0]:inds_foi_amp[1]+1] = True
    if foi_mode=='include':
        cluster_mask[~foi] = False
    elif foi_mode=='exclude':
        cluster_mask[foi] = False
    
fig, axes = plt.subplots(nrows=1, ncols=2, sharey='all', figsize=(4,2), dpi=300)
if zscore:
    data_plot = scipy.stats.zscore(data_orig[:, i_roi, i_fmin_phase:i_fmax_phase+1, 
                     i_fmin_amp:i_fmax_amp+1], axis=None)
else:
    data_plot = data_orig[:, i_roi, i_fmin_phase:i_fmax_phase+1, 
                     i_fmin_amp:i_fmax_amp+1]
vmax = np.max([data_plot[inds_TD].mean(0).max(), 
              abs(data_plot[inds_TD].mean(0).min()),
              data_plot[inds_ASD].mean(0).max(), 
              abs(data_plot[inds_ASD].mean(0).min())]) * 0.8
vmin = -vmax
for ax, inds_group, group in zip(axes, [inds_TD, inds_ASD], ['TD', 'ASD']):
    im = ax.imshow(data_plot[inds_group].mean(0).T, 
                        extent=[fmin_phase, fmax_phase, fmin_amp, fmax_amp],
                        aspect='auto', origin='lower', cmap='jet', # 'RdBu_r'
                        vmin=vmin, vmax=vmax, interpolation='gaussian')
    ax.contour(this_freqs_phase, this_freqs_amp, 
               cluster_mask[i_fmin_phase:i_fmax_phase+1, 
                            i_fmin_amp:i_fmax_amp+1].T, 
               levels=[0.5], colors='k',
               linestyles='dashed', linewidths=1.5, corner_mask=True) 
    ax.set_xticks(np.arange(fmin_phase, fmax_phase+1, 2))
    ax.set_yticks(np.arange(fmin_amp, fmax_amp+1, 20))
    ax.tick_params(axis='both', labelsize=5)
    ax.set_title(group, fontsize=8, fontweight='bold')
    
fig.suptitle('%s' % cond, fontsize=10, x=0.45)
fig.text(.5, .02, 'Phase frequency (Hz)', position=(0.3,0.03), va='center', 
         rotation='horizontal', fontsize=6, fontweight='bold')
fig.text(.02, .5, 'Amplitude frequency (Hz)', ha='center', position=(0.02,0.16),
         rotation='vertical', fontsize=6, fontweight='bold')
plt.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.0325, 0.53])
cbar = fig.colorbar(im, cax=cbar_ax) 
cbar.ax.tick_params(labelsize=5)

fn_out = '%s/figures/pac/pac_%dTD_%dASD_%s_%s_%d-%dms_phase%d-%dHz_amp%d-%dHz_%sReg_cdt%s_pht%s' \
            % (paths['cluster'], n_TD, n_ASD, roi, cond, int(tmin*1e3), int(tmax*1e3), 
                foi_phase[0], foi_phase[1], foi_amp[0], foi_amp[1], covar_out, str(cdt_p), str(pht_p))
fig.savefig('%s.png' % fn_out, dpi=300)
plt.close()

    
#%% within-cluster correlations

roi_id = rois[8]

cluster_data_label = 'MSS-SWS' # which condition to use for clustering, or use readymade clustermask
cdt_p = 0.01
pht_p = None # p-value threshold within-cluster values for a sub mask
cluster_mask_name = ''
correlate_data_labels = ['MSS-Baseline', 'SWS-Baseline', 'MSS-SWS'] # which conditions to use for correlations
exclude = [] # '083701', '093901'
groups = ['TD', 'ASD']

peak = 'max' # 'min', 'max', 'mean', or 'median'
# define frequency range of interest (clusters beyond this range are excluded)
fois_phase = [(8, 12)]
fois_amp = [(30, 100)]
plot = False
save_brains = False
partial = False # regress covariates also from behavioral scores?

score_labels = ['age', 'VIQ', 'ASPS', 'ASPSa1-a5', 'ASPSa1-a7', 'ADOS_tot_old', 'ADOS_comm_soc_old',
                'ICSS', 'SRS_tot_T', 'SRS_socComm_T']
                # 'age', 'ICSS', 'SCSS', 'ASPS', 'ADOS_tot_old', 'ADOS_comm_soc_old',
                #'SRS_tot_T', 'SRS_socComm_T']#, 'NVIQ', 'VIQ', 'age'] # 

c_TD = 'lightgreen'
c_ASD = 'orchid'

for foi_phase, foi_amp in zip(fois_phase, fois_amp):
    print('\n\nPhase %d-%dHz, amplitude %d-%dHz' % (foi_phase[0], foi_phase[1],
                                                    foi_amp[0], foi_amp[1]))
    i_fmin_phase = list(freqs_phase).index(foi_phase[0])
    i_fmax_phase = list(freqs_phase).index(foi_phase[1])
    i_fmin_amp  = list(freqs_amp).index(foi_amp[0])
    i_fmax_amp  = list(freqs_amp).index(foi_amp[1])
        
    
    for roi in [roi for roi in rois if roi_id in roi]:
        print('\nROI: %s' % roi)
        i_roi = rois.index(roi)
        
        inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
        inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
        
        roi_label = roi.split('_')[0]
        roi_label += '-%s' % roi.split('-')[-1]
        
        foi = np.zeros((n_phase_fq, n_amp_fq), dtype=bool)
        foi[i_fmin_phase:i_fmax_phase+1, i_fmin_amp:i_fmax_amp+1] = True
        
        if cluster_data_label:
            cluster_data = data_dict[cluster_data_label]
            # th = dict(start=0, step=0.2)
            th_p = cdt_p
            th = scipy.stats.distributions.f.ppf(1. - th_p / 2., n_ASD - 1, n_TD - 1)
            n_perms = 5000
            X = [cluster_data[inds_TD,i_roi], cluster_data[inds_ASD,i_roi]]
            print('\nTD vs. ASD permutation cluster test: %s' % cluster_data_label)
            stats, clusters, cluster_pvals, _ = mne.stats.permutation_cluster_test(
                X, threshold=th, n_jobs=12, out_type='mask', 
                n_permutations=n_perms+1, seed=7)
            good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
            print('Min cluster p-val: %.3f\n' % min(cluster_pvals))
            cluster_mask = np.zeros((n_phase_fq, n_amp_fq), dtype=bool)
            if good_cluster_inds.size > 0:
                stats_th = np.zeros(stats.shape)
                if type(th)==dict:
                    stats_th[good_cluster_inds] = \
                        stats[good_cluster_inds]
                else:
                    for good_cluster_ind in good_cluster_inds:
                        stats_th[clusters[good_cluster_ind]] = \
                            stats[clusters[good_cluster_ind]]
                if pht_p:
                    pht_F = scipy.stats.distributions.f.ppf(1. - pht_p / 2., 
                                                               n_ASD - 1, n_TD - 1)
                    stats_th[np.where(stats_th < pht_F)] = 0
                    
                cluster_mask[np.nonzero(stats_th)] = True
                cluster_mask[~foi] = False
                # plt.imshow(stats_th.T, extent=[freqs_phase[0], freqs_phase[-1], 
                #                                     freqs_amp[0], freqs_amp[-1]],
                #                 aspect='auto', origin='lower', cmap='RdBu_r', 
                #                 interpolation='gaussian')
                # np.save('%s/clusterMask_%s_%s.npy' % (paths['cluster'], 
                #                                        cluster_data_label, roi),
                #                                         cluster_mask)
        else:
            cluster_mask = np.load('%s/%s.npy' % (paths['cluster'], cluster_mask_name))
            cluster_mask[~foi] = False
            
        if np.any(cluster_mask):
            for correlate_data_label in correlate_data_labels:
                correlate_data = data_dict[correlate_data_label]
                print('\nCalculating correlation for %s' % correlate_data_label)
                
                if save_brains:
                    brains = correlate_data[:, i_roi, cluster_mask].mean(-1)
                    # inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
                    # inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
                    np.save('%s/npy/pac_N%d_%s_%s_phase%d-%dHz_amp%d-%dHz_%sReg_cdt%s_ph%s' 
                            % (paths['cluster'], n_subs, roi_label, correlate_data_label,  
                               foi_phase[0], foi_phase[1], foi_amp[0], foi_amp[1],
                               covar_out, str(cdt_p), str(pht_p)), brains)
                    # np.save('%s/npy/%dASD_pac_%s_%s_phase%d-%dHz_amp%d-%dHz_%sReg_cdt%s_ph%s' 
                    #         % (paths['cluster'], n_ASD, roi_label, correlate_data_label,  
                    #            foi_phase[0], foi_phase[1], foi_amp[0], foi_amp[1],
                    #            covar_out, str(cdt_p), str(ph_p)), brains[inds_ASD])
                    
                for score_label in score_labels:
                    if peak=='min':
                        brains = correlate_data[:, i_roi, cluster_mask].min(-1)
                    elif peak=='max':
                        brains = correlate_data[:, i_roi, cluster_mask].max(-1)
                    elif peak=='mean':
                        brains = correlate_data[:, i_roi, cluster_mask].mean(-1)
                    elif peak=='median':
                        brains = np.median(correlate_data[:, i_roi, cluster_mask], axis=-1)
                    scores = np.array(sub_info[score_label])
                    diagnosed = sub_info['ASD'].copy()
                    this_sub_IDs = sub_info['sub_ID'].copy()
                    
                    # indices of None, i.e. missing scores
                    null_inds = [i for i in range(len(scores)) 
                                 if scores[i]==None]
                    null_inds += [sub_IDs.index(sub_ID) for sub_ID in exclude 
                                  if sub_IDs.index(sub_ID) not in null_inds]
                    null_inds = list(np.sort(null_inds))
                    null_inds.reverse() # reverse so that largest index is removed first
                    # remove Nones and the corresponding brain data
                    for i in null_inds:
                        scores = np.delete(scores, i)
                        brains = np.delete(brains, i, axis=0)
                        diagnosed.pop(i)   
                        this_sub_IDs.pop(i)
                        
                    if partial and covar_labels:
                        #% Regress covariates from scores
                        covar_out = '%s' % ('').join(covar_labels)
                        covars = []
                        for covar_label in covar_labels:
                            covars.append(sub_info[covar_label])
                        for i in null_inds: 
                            covars = np.delete(covars, i, axis=1)
                        model = np.zeros((len(this_sub_IDs),len(covars)+1))
                        model[:,0] = np.ones(len(this_sub_IDs))
                        model[:,1::] = np.transpose(covars)
                           
                        beta = scipy.linalg.lstsq(model, list(scores))[0]
                        scores = list(scores) - model.dot(beta)
                        
                        # zscore bscores
                        scores = scipy.stats.zscore(scores, axis=None)
                    if covar_labels:
                        brains = scipy.stats.zscore(brains, axis=None)
                        
                    inds_TD = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
                    inds_ASD = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
                           
                    fn_out = '%s/figures/pac/correlations/%dTD_%dASD_cluster_%s_%s_%s_phase%d-%dHz_amp%d-%dHz' \
                                % (paths['cluster'], n_TD, n_ASD, roi_label, correlate_data_label,  
                                   score_label, foi_phase[0], foi_phase[1], foi_amp[0], foi_amp[1],)
                    if exclude:
                        fn_out += '_%s_excluded' % ('-').join(exclude)
                    fn_out += '_%sReg_cdt%s_ph%s_%s' % (covar_out, str(cdt_p), 
                                                        str(pht_p), peak)
                    # if cluster_data_label:
                    #     fn_out += '_%s' % cluster_data_label
                    # else:
                    #     fn_out += '_%s' % cluster_mask_name
                    
                    if 'all' in groups:
                        p_all = 1
                        r_all, p_all = pearsonr(scores, brains)
                        if p_all < 0.1:
                            print('\nAll: %s vs. %s in %s: r=%f, p=%f' 
                                  % (correlate_data_label, score_label, roi_label, r_all, p_all))                    

                    if 'TD' in groups:
                        p_TD = 1
                        if len(scores[inds_TD])>0:
                            r_TD, p_TD = pearsonr(scores[inds_TD], brains[inds_TD])
                            if p_TD < 0.1:
                                print('\nTD: %s vs. %s in %s: r=%f, p=%f' 
                                      % (correlate_data_label, score_label, roi_label, r_TD, p_TD))  
                                
                    if 'ASD' in groups:
                        p_ASD = 1
                        if len(scores[inds_ASD])>0:
                            r_ASD, p_ASD = pearsonr(scores[inds_ASD], brains[inds_ASD])
                            if p_ASD < 0.1:
                                print('\nASD: %s vs. %s in %s: r=%f, p=%f' 
                                      % (correlate_data_label, score_label, roi_label, r_ASD, p_ASD))
                                
                    if 'TD' in groups and 'ASD' in groups:            
                        p = 1
                        if len(scores[inds_TD])>0 and len(scores[inds_ASD])>0:
                            z,p = compare_corr_coefs(r_TD, r_ASD, len(inds_TD), len(inds_ASD))
                            if p<0.05:
                                print('\nTD vs. ASD: %s vs. %s in %s: z=%f, p=%f' 
                                      % (correlate_data_label, score_label, roi_label, z, p))
                        
                    if plot and (p_TD<0.05 or p_ASD<0.05 or p<0.05):
                        plt.figure()
                        if 'TD' in groups and len(scores[inds_TD])>0:
                            ax = sns.regplot(list(scores[inds_TD]), brains[inds_TD], ci=None, 
                                        scatter_kws={'s':80}, color=c_TD, marker='o')
                            ax.text(0.025, 0.925, 'r=%.2f, p=%.3f' % (r_TD, p_TD), fontsize=12,                                    
                                    color=c_TD, transform=ax.transAxes)
                        if 'ASD' in groups and len(scores[inds_ASD])>0:
                            ax = sns.regplot(list(scores[inds_ASD]), brains[inds_ASD], ci=None, 
                                        scatter_kws={'s':80}, color=c_ASD, marker='s')
                            ax.text(0.025, 0.875, 'r=%.2f, p=%.3f' % (r_ASD, p_ASD), fontsize=12,                                    
                                    color=c_ASD, transform=ax.transAxes)
                            for i, i_ASD in enumerate(inds_ASD):
                                ax.annotate(this_sub_IDs[i_ASD], (scores[inds_ASD][i], brains[inds_ASD][i]))
                            
                        if len(scores[inds_TD])>0 and len(scores[inds_ASD])>0:
                            if 'TD' in groups and 'ASD' in  groups:
                                ax.text(0.025, 0.825, 'z=%.2f, p=%.3f' % (z, p), fontsize=12,                                    
                                        color='k', transform=ax.transAxes)    
                                
                            if 'all' in groups:
                                plt.figure()
                                ax = sns.regplot(list(scores), brains, ci=None, 
                                        scatter_kws={'s':10}, color='k', marker='o')
                                ax = sns.scatterplot(list(scores[inds_TD]), brains[inds_TD], ci=None, 
                                        color=c_TD, marker='o', s=80)
                                ax = sns.scatterplot(list(scores[inds_ASD]), brains[inds_ASD], ci=None, 
                                        color=c_ASD, marker='o', s=80)
                                ax.text(0.025, 0.925, 'r=%.2f, p=%.3f' % (r_all, p_all), fontsize=12,                                    
                                        color='k', transform=ax.transAxes)
                                # slope,intercept = np.polyfit(list(scores), list(brains), 1)
                                # plt.plot(list(scores), slope*scores + intercept, 'k-')
                            
                        ax.set_xlim(min(scores)-(max(scores)-min(scores))*0.05, 
                                    max(scores)+(max(scores)-min(scores))*0.05)
                        ax.set_ylim(min(brains)-(max(brains)-min(brains))*0.05, 
                                    max(brains)+(max(brains)-min(brains))*0.05)
                        plt.title('%s comod cluster %d-%dHz phase, %d-%dHz amp' 
                                  % (roi_label, foi_phase[0], foi_phase[1], 
                                     foi_amp[0], foi_amp[1]))
                        plt.xlabel('%s' % score_label)
                        plt.ylabel('%s' % correlate_data_label) 
                        plt.tight_layout()
                        plt.savefig('%s.png' % fn_out, dpi=300)
                        plt.close('all')
                    
                        
                        

#%% within-cluster bar plots

roi_id = 'cdt0.05_phNone_dSPM_5verts_MSS_SWS_peak0-500ms_restrictedParietal_auto-lh'
cluster_data_label = 'MSS-SWS' # which condition to use for clusters, or load cluster mask
cluster_mask_name = ''

cdt_p = 0.05
pht_p = None # threshold within-cluster values for a sub mask

plot_data_labels = ['MSS-Baseline', 'SWS-Baseline', 'MSS-SWS']
hemi = 'lh'
peak = 'mean'
# define frequency range of interest (clusters beyond this range are excluded)
fois_phase = [(8, 14)]#, (9, 11), (12, 14)]
fois_amp = [(30, 100)]#, (30, 50), (80, 100)]
plot = True
nonparametric = True # two-sample t-test of nonparametric wilcoxon rank sums

inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']

group_labels = sub_info['ASD']*len(plot_data_labels)
# replace 'Yes' with 'ASD' and 'No' with 'TD' in group labels
group_labels = ['ASD' if group_labels[i]=='Yes' else 'TD' 
                for i in range(len(group_labels))]

c_TD = 'lightgreen'
c_ASD = 'orchid'

for foi_phase, foi_amp in zip(fois_phase, fois_amp):
    print('\n\nPhase %d-%dHz, amplitude %d-%dHz' % (foi_phase[0], foi_phase[1],
                                                  foi_amp[0], foi_amp[1]))
    i_fmin_phase = list(freqs_phase).index(foi_phase[0])
    i_fmax_phase = list(freqs_phase).index(foi_phase[1])
    i_fmin_amp  = list(freqs_amp).index(foi_amp[0])
    i_fmax_amp  = list(freqs_amp).index(foi_amp[1])
    
    for roi in [roi for roi in rois if roi_id in roi]:
        print('\nROI: %s' % roi)
        i_roi = rois.index(roi)    
        roi_label = roi.split('_')[0]
        roi_label += '-%s' % roi.split('-')[-1]
        
        foi = np.zeros((n_phase_fq, n_amp_fq), dtype=bool)
        foi[i_fmin_phase:i_fmax_phase+1, i_fmin_amp:i_fmax_amp+1] = True
        
        if cluster_data_label:
            fn_out = '%s/figures/pac/pac_barPlot_%dTD_%dASD_%s_cluster_phase%d-%dHz_amp%d-%dHz_%s' \
                                    % (paths['cluster'], n_TD, n_ASD, roi_label,   
                                       foi_phase[0], foi_phase[1], foi_amp[0], foi_amp[1], 
                                       ('_').join(plot_data_labels))
            if nonparametric:
                fn_out += '_ranksum'
            else:
                fn_out += '_ttest'
            fn_out += '_%sReg_cdt%s_ph%s' % (covar_out, str(cdt_p), str(pht_p))
            
            cluster_data = data_dict[cluster_data_label]
            # th = dict(start=0, step=0.2)
            th_p = cdt_p
            th = scipy.stats.distributions.f.ppf(1. - th_p / 2., n_ASD - 1, n_TD - 1)
            n_perms = 5000
            X = [cluster_data[inds_TD,i_roi], cluster_data[inds_ASD,i_roi]]
            print('\nTD vs. ASD permutation cluster test: %s' % cluster_data_label)
            stats, clusters, cluster_pvals, _ = mne.stats.permutation_cluster_test(
                X, threshold=th, n_jobs=12, out_type='mask', 
                n_permutations=n_perms+1, seed=7)
            good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
            cluster_mask = np.zeros((n_phase_fq, n_amp_fq), dtype=bool)
            if good_cluster_inds.size > 0:
                stats_th = np.zeros(stats.shape)
                if type(th)==dict:
                    stats_th[good_cluster_inds] = \
                        stats[good_cluster_inds]
                else:
                    for good_cluster_ind in good_cluster_inds:
                        stats_th[clusters[good_cluster_ind]] = \
                            stats[clusters[good_cluster_ind]]
                            
                if pht_p:
                    pht_F = scipy.stats.distributions.f.ppf(1. - pht_p / 2., 
                                                               n_ASD - 1, n_TD - 1)
                    stats_th[np.where(stats_th < pht_F)] = 0
                        
                cluster_mask[np.nonzero(stats_th)] = True
                cluster_mask[~foi] = False
        else:
            cluster_mask = np.load('%s/%s.npy' % (paths['cluster'], cluster_mask_name))
            cluster_mask[~foi] = False
            
        if np.any(cluster_mask):
            data, stat, p = [], [], []
            for plot_data_label in plot_data_labels:
                if peak=='min':
                    data_temp = data_dict[plot_data_label][:, i_roi, cluster_mask].min(-1)
                elif peak=='max':
                    data_temp = data_dict[plot_data_label][:, i_roi, cluster_mask].max(-1)
                elif peak=='mean':
                    data_temp = data_dict[plot_data_label][:, i_roi, cluster_mask].mean(-1)
                elif peak=='median':
                    data_temp = np.median(data_dict[plot_data_label][:, i_roi, cluster_mask], axis=-1)
                
                data.append(data_temp)
                if nonparametric:
                    temp = ranksums(data_temp[inds_TD], data_temp[inds_ASD])
                else:
                    temp = ttest_ind(data_temp[inds_TD], data_temp[inds_ASD])
                stat.append(temp[0])
                p.append(temp[1])
    
            if covar_labels:
                data = scipy.stats.zscore(np.concatenate(data))
            else:
                data = np.concatenate(data)
                
            data_pd = {'PAC': data, 
                       'Group': group_labels,
                       'Condition': np.repeat(plot_data_labels, n_subs).tolist(),
                       'ID': sub_info['sub_ID']*len(plot_data_labels)}    
            data_frame = pd.DataFrame(data=data_pd)
            
            plt.figure()
            ax = sns.barplot(x='Condition', y='PAC', hue='Group', data=data_frame,
                              alpha=1, edgecolor='gray', ci=68, palette=[c_TD, c_ASD],
                              hue_order=['TD', 'ASD'])
            plt.title('%s comod cluster phase %d-%dHz, amp %d-%dHz' 
                      % (roi_label, foi_phase[0], foi_phase[1], foi_amp[0], foi_amp[1]))
            plt.legend(loc='lower left')
            signifs = []
            for val in p:
                if val>0.06: signifs.append('')
                elif val<0.06 and val>0.05: signifs.append('(*)')
                elif val<0.05 and val>0.01: signifs.append('*')
                elif val<0.01 and val>0.001: signifs.append('**')
                elif val<0.001: signifs.append('***')
                
            yloc = ax.get_ybound()[1] - (ax.get_ybound()[1] * 0.15)
            for signif, xloc in zip(signifs, ax.get_xticks()):
                ax.text(xloc, yloc, signif, fontsize=20,
                        horizontalalignment='center') 
            plt.tight_layout()
            plt.savefig('%s.png' % fn_out, dpi=300)
            plt.close()

    
    
#%% Bar plots

cois = ['Speech-Baseline', 'Jabber-Baseline', 'Speech-Jabber']
roi_id = 'manual'
foi_phase = (8, 12)
foi_amp = (60, 100)
tw_labels = ['0-500ms', '500-1000ms', '1000-1500ms']

i_fmin_phase = list(freqs_phase).index(foi_phase[0])
i_fmax_phase = list(freqs_phase).index(foi_phase[1])
i_fmin_amp  = list(freqs_amp).index(foi_amp[0])
i_fmax_amp  = list(freqs_amp).index(foi_amp[1])

group_labels = sub_info['ASD']*len(cois)*len(data_all)
# replace 'Yes' with 'ASD' and 'No' with 'TD' in group labels
group_labels = ['ASD' if group_labels[i]=='Yes' else 'TD' 
                for i in range(len(group_labels))]

c_TD = 'lightgreen'
c_ASD = 'orchid'


for roi in [roi for roi in rois if roi_id in roi]:
    i_roi = rois.index(roi)
    roi_label = roi.split('_')[0]
    roi_label += '-%s' % roi.split('-')[-1]
    for inds_group,group in zip([inds_TD, inds_ASD], ['TD', 'ASD']):
        data = []
        for this_data in data_all: # time windows
            for coi in cois: # conditions        
                data.append(this_data[coi][inds_group, i_roi, i_fmin_phase:i_fmax_phase,
                                           i_fmin_amp:i_fmax_amp].mean((-1,-2)))
    
        data_pd = {'PAC': np.concatenate(data), 
                   # 'Group': group_labels,
                   'Condition': np.repeat(cois, len(inds_group)).tolist() * len(data_all),
                   'Time window': np.repeat(tw_labels, len(inds_group) * len(cois)),
                   'ID': inds_group*len(cois)*len(data_all)}    
        data_frame = pd.DataFrame(data=data_pd)
        
        plt.figure()
        sns.barplot(x='Time window', y='PAC', hue='Condition', data=data_frame,
                      alpha=1, edgecolor='gray', ci=68) # , palette=[c_ASD,c_TD]
        plt.title('%s - %s' % (roi_label, group))

#%% Window analysis

# time/freq limits for plotting
fwins_phase = [(7, 11)] # (4, 14), 
fwins_amp = [(30, 60)] # (30, 100), 

score_labels = ['age', 'NVIQ', 'VIQ', 'ICSS', 'SCSS', 'ASPS', 'SRS_tot_T'] # 'age', 'ICSS', 'SCSS', 'ASPS', 'SRS_tot_T'
plot = False

for score_label in score_labels:
    for this_data,data_label in zip(data, data_labels):
        for i_roi,roi in enumerate(rois):
            if roi.startswith('AC'):
                roi_label = 'AC'
            elif roi.startswith('IP'):
                roi_label = 'IP' 
            elif roi.startswith('IF'):
                roi_label = 'IF'
            if 'manual' in roi:
                roi_label += '_manual'
            else:
                roi_label += '_auto'
            if roi.endswith('lh'):
                roi_label += '-lh'
            else:
                roi_label += '-rh'        
                
            for fwin_phase in fwins_phase:
                fwin_phase_label = '%d-%dHz' % (fwin_phase[0], fwin_phase[1])
                i_fmin_phase = list(freqs_phase).index(fwin_phase[0])
                i_fmax_phase = list(freqs_phase).index(fwin_phase[1])
                for fwin_amp in fwins_amp:
                    fwin_amp_label = '%d-%dHz' % (fwin_amp[0], fwin_amp[1])
                    i_fmin_amp = list(freqs_amp).index(fwin_amp[0])
                    i_fmax_amp = list(freqs_amp).index(fwin_amp[1])
                        
                    brains = this_data[:,i_roi,i_fmin_phase:i_fmax_phase,
                                       i_fmin_amp:i_fmax_amp].mean((-1,-2))
                    scores = np.array(sub_info[score_label])
                    diagnosed = sub_info['ASD'].copy()
                    
                    # indices of None, i.e. missing scores
                    null_inds = [i for i in range(len(scores)) 
                                 if scores[i]==None]
                    null_inds.reverse() # reverse so that largest index is removed first
                    # remove Nones and the corresponding brain data
                    for i in null_inds:
                        scores = np.delete(scores, i)
                        brains = np.delete(brains, i)
                        diagnosed.pop(i)   
                        
                    inds_TD = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
                    inds_ASD = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
                    for group_label,inds_group,group_color in zip(['TD', 'ASD'], 
                                                                  [inds_TD, inds_ASD], 
                                                                  ['lightgreen', 'orchid']):
                        
                        if not len(scores[inds_group])==0:
                            r,p = pearsonr(scores[inds_group], brains[inds_group])
                            if p<0.05 and data_label=='MSS-SWS':
                                print('\n%s - %s vs. %s, %s, %s in %s: r=%f, p=%f' 
                                      % (group_label, data_label, score_label, fwin_phase_label, 
                                         fwin_amp_label, roi_label, r, p))
            
                                if plot:
                                    plt.figure()
                                    ax = sns.regplot(list(scores[inds_ASD]), brains[inds_ASD], ci=None, 
                                                scatter_kws={'s':80}, color='orchid', marker='s')
                                    if len(scores[inds_TD])>0:
                                        sns.regplot(list(scores[inds_TD]), brains[inds_TD], ci=None, 
                                                    scatter_kws={'s':80}, color='lightgreen', marker='o')
                                    
                                    ax.set_xlim(min(scores)-(max(scores)-min(scores))*0.05, 
                                                max(scores)+(max(scores)-min(scores))*0.05)
                                    ax.set_ylim(min(brains)-(max(brains)-min(brains))*0.05, 
                                                max(brains)+(max(brains)-min(brains))*0.05)
                                    plt.title('%s - %s - %s' % (roi_label, fwin_phase_label, fwin_amp_label))
                                    plt.xlabel('%s' % score_label)
                                    plt.ylabel('%s' % data_label) 
                                    plt.tight_layout()
                                    plt.savefig('%s/figures/pac/correlations/%s_%s_%s_%s_%s.png' 
                                                % (paths['cluster'], data_label, score_label,
                                                   roi_label, fwin_phase_label, fwin_amp_label), dpi=300)
                                    plt.close()
                                    
                                
    
