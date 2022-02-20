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
import random
import sys
import multiprocessing
import os
import matplotlib.pyplot as plt
import scipy
from scipy.stats import mannwhitneyu, pearsonr
import seaborn as sns

#% set up
paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# define bad subjects
bad_subs = ['105801', '107301']
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

fn_in = 'induced_-200-2000ms_4-100Hz_MSS-SWS_dSPMfROIsManual0-500ms.p'

# load one to get variables
f = open('%s/%s/p/%s' % (paths['cluster'], sub_IDs[0], fn_in), 'rb')
data = pickle.load(f)
f.close()

conds = data['conds']
n_conds = len(conds)
rois = data['rois']
n_rois = len(rois)
times = data['times']
n_times = len(times)
freqs = list(data['freqs'])
n_freqs = len(freqs) 


#%% Load data

data = np.zeros((n_subs, n_conds, n_rois, n_freqs, n_times))
for i_sub,sub_ID in enumerate(sub_IDs):
    print(sub_ID)
    # subject path
    sub_path = '%s/%s/' % (paths['cluster'], sub_ID)
    
    #% load data
    try:
        f = open('%s/p/%s' % (sub_path, fn_in), 'rb')
        temp = pickle.load(f)
        f.close()
    except:
        print('Could not find %s for %s' % (fn_in, sub_ID))
        continue
    
    for i_cond,cond in enumerate(conds):
        data[i_sub,i_cond] = temp['induced_power'][cond]
data_orig = data        
#%% Plot & report

if 'Manual' in fn_in: 
    out_id = 'fROIsManual'
else:
    out_id = 'fROIsAuto'

hemi = 'lh'

# time/freq limits for plotting
tmin = 0. # in s
tmax = 1.99
tstep = 0.01
fmin = 60 # in Hz
fmax = 100
i_tmin = list(times).index(tmin)
i_tmax = list(times).index(tmax)
i_fmin = list(freqs).index(fmin)
i_fmax = list(freqs).index(fmax)
i_tstep = list(times).index(tstep)-list(times).index(0)

fontsize = 5

TD_IDs = [sub_IDs[i] for i in inds_TD]
ASD_IDs = [sub_IDs[i] for i in inds_ASD]
TD_ages = [sub_info['age'][i] for i in inds_TD]
ASD_ages = [sub_info['age'][i] for i in inds_ASD]
inds_TD_ageSorted = np.argsort(TD_ages)
inds_ASD_ageSorted = np.argsort(ASD_ages)
TD_IDs_ageSorted = [TD_IDs[i] for i in inds_TD_ageSorted]
ASD_IDs_ageSorted = [ASD_IDs[i] for i in inds_ASD_ageSorted]
sub_IDs_sorted = TD_IDs_ageSorted + ASD_IDs_ageSorted

data_contrast = data[:,0,:,i_fmin:i_fmax,i_tmin:i_tmax:i_tstep]-\
                data[:,1,:,i_fmin:i_fmax,i_tmin:i_tmax:i_tstep]
n_times = data_contrast.shape[-1]
n_freqs = data_contrast.shape[-2]
              
data_IDs = ['TD group mean', 'ASD group mean', 
            'TD vs. ASD t-test', 
            'TD vs. ASD t-test at p=.05',
            'TD vs. ASD permutation cluster test at p=.05']

score_labels = ['age', 'NVIQ', 'VIQ', 'ICSS', 'SCSS', 'ASPS', 'SRS_tot_T', 
                'SRS_socComm_T', 'ADOS_tot_new', 'ADOS_comm_soc_new']
data_IDs += ['%s_%s' % (group, score_label) for score_label in score_labels 
             for group in ['TD', 'ASD']]
                
report = mne.report.Report()
figs = []
captions = []
for data_ID in sub_IDs_sorted + data_IDs:
    if data_ID.isdigit():  
        i_sub = sub_info['sub_ID'].index(data_ID)
        age = sub_info['age'][sub_info['sub_ID'].index(data_ID)]

        if sub_info['ASD'][sub_info['sub_ID'].index(data_ID)]=='Yes':
            group_label = 'ASD'
        else:
            group_label = 'TD'
        captions.append('%s - %.1f yo %s' % (data_ID, age, group_label))
    else:
        captions.append(data_ID)
        
    if hemi=='bihemi':
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all', 
                                 figsize=(4,6), dpi=300)
        this_rois = rois
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', 
                                 figsize=(8,2), dpi=300)
        if hemi=='lh':
            this_rois = [roi for roi in rois if 'lh' in roi]
        else:
            this_rois = [roi for roi in rois if 'rh' in roi]
        
    for i_roi,roi in enumerate(this_rois):
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
            
        if data_ID.isdigit():  
            plot_data = data_contrast[i_sub,i_roi]
            vmin = -8
            vmax = 8
        elif data_ID=='TD group mean':
            plot_data = data_contrast[inds_TD,i_roi].mean(0)
            vmin = np.min(data_contrast.mean(0)) * 0.8
            vmax = np.max(data_contrast.mean(0)) * 0.8
        elif data_ID=='ASD group mean':
            plot_data = data_contrast[inds_ASD,i_roi].mean(0)
            vmin = np.min(data_contrast.mean(0)) * 0.8
            vmax = np.max(data_contrast.mean(0)) * 0.8
        elif 't-test' in data_ID:
            t,p = scipy.stats.ttest_ind(data_contrast[inds_TD,i_roi],
                                        data_contrast[inds_ASD,i_roi])
            if 'at p=.05' in data_ID:
                t_th = np.zeros(t.shape)
                t_th[np.where(p<0.05)] = t[np.where(p<0.05)]
                plot_data = t_th
                vmin = -2
                vmax = 2                    
            else:
                plot_data = t
                vmax = plot_data.max() * 0.8
                vmin = plot_data.min() * 0.8
                
        elif 'permutation cluster' in data_ID:
            th = dict(start=0, step=0.2)
            n_perms = 1000
            X = [data_contrast[inds_TD,i_roi], data_contrast[inds_ASD,i_roi]]
            stats, clusters, cluster_pvals, _ = mne.stats.permutation_cluster_test(
                X, threshold=th, n_jobs=12, out_type='indices', n_permutations=n_perms+1)
            good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
            if good_cluster_inds.size > 0:
                stats_th = np.zeros(stats.size)
                stats_th[good_cluster_inds] = \
                    stats.reshape(stats.shape[0] * stats.shape[1])[good_cluster_inds]
                plot_data = np.reshape(stats_th, (stats.shape[0], stats.shape[1]))
                vmax = stats_th.max() * 0.8
                vmin = -vmax
            else:
                plot_data = np.zeros(stats.shape)
                vmin = -1
                vmax = 1
            
        else: # correlations
            brains = data_contrast[:, i_roi].reshape(n_subs, n_times*n_freqs)
            scores = np.array(sub_info[('_').join(data_ID.split('_')[1::])])
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
                
            if 'TD' in data_ID:
                inds_group = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
            else:
                inds_group = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
                               
            if not len(scores[inds_group])==0:
                r = np.zeros((n_freqs*n_times))
                p = np.zeros((n_freqs*n_times))
                for i in np.arange(n_freqs*n_times):
                    r[i],p[i] = pearsonr(scores[inds_group], brains[inds_group, i])
                r = r.reshape((n_freqs, n_times))
                p = p.reshape((n_freqs, n_times))
                
                plot_data = r
            else:
                plot_data = np.zeros((n_times, n_freqs))
            
        im = ax.imshow(plot_data, extent=[tmin, tmax, fmin, fmax],
                       aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r')
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(vmin,vmax,2))
        cbar.ax.tick_params(labelsize=fontsize)
        ax.set_title('%s' % roi_label, fontsize=fontsize)
        ax.set_xticks(np.arange(tmin,tmax,0.2))
        ax.set_yticks(np.arange(fmin,fmax,5))
        ax.tick_params(axis='both', labelsize=fontsize)

    fig.text(.5, .02, 'Time (s)', ha='center', va='center', 
             rotation='horizontal', fontsize=fontsize)
    fig.text(.02, .5, 'Frequency (Hz)', ha='center', va='center', 
             rotation='vertical', fontsize=fontsize)
    plt.tight_layout()
        
    figs.append(fig)
    plt.close()
  
report.add_figs_to_section(figs, captions=captions)
report.save('%s/induced_power_MSS-SWS_%d-%dms_%d-%dHz_%s_%s.html' 
            % (paths['cluster'], int(tmin*1000), int(tmax*1000), fmin, fmax, 
               hemi, out_id), overwrite=True, open_browser=False)


#%% Window analysis

# time/freq limits for plotting
twins = [(0., 0.8), (0.8, 1.6)] #, '800-1600ms':(0.8, 1.6)} # in s
fwins = [(40, 60), (80, 100)]#, '80-100Hz':(80, 100)} # in Hz

score_label = 'age'
plot = False

scores = np.array(sub_info[score_label])
for i_roi,roi in enumerate(rois):
    if roi.startswith('AC'):
        roi_label = 'AC'
    elif roi.startswith('IP'):
        roi_label = 'IP' 
    elif roi.startswith('IF'):
        roi_label = 'IF'
    if roi.endswith('lh'):
        roi_label += '-lh'
    else:
        roi_label += '-rh'        
        
    for twin in twins:
        twin_label = '%d-%dms' % (int(twin[0]*1000), int(twin[1]*1000))
        for fwin in fwins:
            fwin_label = '%d-%dHz' % (fwin[0], fwin[1])
            
            i_tmin = list(times).index(twin[0])
            i_tmax = list(times).index(twin[1])
            i_fmin = list(freqs).index(fwin[0])
            i_fmax = list(freqs).index(fwin[1])
        
            data_contrast = data[:,0,i_roi,i_fmin:i_fmax,i_tmin:i_tmax]-\
                            data[:,1,i_roi,i_fmin:i_fmax,i_tmin:i_tmax]
            if plot: plt.figure()
            for group_label,inds_group,group_color in zip(['TD','ASD'], 
                                                          [inds_TD,inds_ASD], 
                                                          ['lightgreen', 'orchid']):
                r,p = scipy.stats.pearsonr(scores[inds_group], 
                                           data_contrast[inds_group].mean((-1,-2)))
                print('\n%s - MSS-SWS vs. %s, %s, %s in %s: r=%f, p=%f' 
                      % (group_label, score_label, twin_label, fwin_label, roi_label, r, p))

                if plot:
                    sns.regplot(scores[inds_group], data_contrast[inds_group].mean((-1,-2)), 
                                ci=None, scatter_kws={'s':80}, color=group_color, 
                                marker='o')
                    sns.regplot(scores[inds_ASD], data_contrast[inds_ASD].mean((-1,-2)), 
                                ci=None, scatter_kws={'s':80}, color=group_color, marker='o')
                    plt.title('%s - %s - %s' % (roi_label, twin_label, fwin_label))
                    plt.xlabel('age')
                    plt.ylabel('MSS-SWS')


#%% Regress covariates out from the data

covar_labels = ['age', 'NVIQ', 'VIQ']

print('Regressing out covariates: %s' % covar_labels)
covars = []
for covar_label in covar_labels:
    covars.append(sub_info[covar_label])
model = np.zeros((n_subs,len(covars)+1))
model[:,0] = np.ones(n_subs)
model[:,1::] = np.transpose(covars)
   
data = np.reshape(data_orig, (n_subs, n_conds*n_rois*n_freqs*n_times))
beta = scipy.linalg.lstsq(model, data)[0]
data = data - model.dot(beta)
data = np.reshape(data, (n_subs, n_conds, n_rois, n_freqs, n_times))
    

#%% permutation cluster test
        
data_conds = [data[:,0], data[:,1], data[:,0]-data[:,1]]
labels_conds = conds + ['%s-%s' % (conds[0], conds[1])]
threshold = dict(start=0, step=1) # TFCE
# p_threshold = 0.01
# threshold = scipy.stats.distributions.f.ppf(1. - p_threshold / 2., n_ASD - 1, n_TD - 1)
n_perms = 1000
tstep = 10 # in ms

print('Permutation cluster test: TD vs. ASD')
for data_cond,cond in zip(data_conds, labels_conds):
    for i_roi,roi in enumerate(rois):
        X = [data_cond[inds_TD,i_roi,:,0:-1:tstep], data_cond[inds_ASD,i_roi,:,0:-1:tstep]]
        stats, clusters, cluster_pvals, _ = mne.stats.permutation_cluster_test(
            X, threshold=threshold, n_jobs=12, out_type='indices', n_permutations=n_perms+1)
        
        good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
        print('%s - %s - smallest cluster p-value: %.2f\n' % (cond, roi, min(cluster_pvals)))
        
        # if good_cluster_inds.size > 0:
        #     stats_th=np.zeros(stats.size)
        #     stats_th[good_cluster_inds]=stats.reshape(stats.shape[0]*stats.shape[1])[good_cluster_inds]
        #     stats_th = np.reshape(stats_th, (stats.shape[0],stats.shape[1]))



#%% t-tests
        
data_conds = [data[:,0], data[:,1], data[:,0]-data[:,1]]
labels_conds = conds + ['%s-%s' % (conds[0], conds[1])]

for data_cond,cond in zip(data_conds, labels_conds):
    for i_roi,roi in enumerate(rois):
        data_TD = data_cond[inds_TD,i_roi]
        data_ASD = data_cond[inds_ASD,i_roi]
        if cond=='MSS-SWS':
            for group_data,group in zip([data_TD, data_ASD], ['TD', 'ASD']):
                t,p = scipy.stats.ttest_1samp(group_data, 0)
                plt.imshow(t, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       aspect='auto', origin='lower', cmap='RdBu_r')
                plt.xlabel('Time (ms)')
                plt.ylabel('Frequency (Hz)')
                plt.title('%s' % cond)
                plt.colorbar()
                plt.savefig('%s/figures/misc/induced_power/%s_%s_oneSampleT_%s.png' 
                            % (paths['cluster'], group, cond, roi), dpi=300)
                plt.close()
                
                # thresholding
                t_th = np.zeros(t.shape)
                t_th[np.where(p<0.05)] = t[np.where(p<0.05)]
                plt.imshow(t_th, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                           aspect='auto', origin='lower', cmap='RdBu_r')
                plt.xlabel('Time (ms)')
                plt.ylabel('Frequency (Hz)')
                plt.title('%s' % cond)
                plt.colorbar()
                plt.savefig('%s/figures/misc/induced_power/%s_%s_oneSampleT_th05_%s.png' 
                            % (paths['cluster'], group, cond, roi), dpi=300)
            
            
        
#%% mann whitney u-tests

data_conds = [data[:,0]-data[:,1]] # data[:,0], data[:,1], 
labels_conds = ['%s-%s' % (conds[0], conds[1])] # conds + 

for data_cond,cond in zip(data_conds, labels_conds):
    for i_roi,roi in enumerate(rois):
        
        u1 = np.zeros(n_freqs*n_times)
        p = np.zeros(n_freqs*n_times)
        data_TD = data_cond[inds_TD,i_roi].reshape(n_TD,n_freqs*n_times)
        data_ASD = data_cond[inds_ASD,i_roi].reshape(n_ASD,n_freqs*n_times)
        for i in range(n_freqs*n_times):
            print('%d/%d' %(i,n_freqs*n_times))        
            u1[i],p[i] = mannwhitneyu(data_TD[:,i], data_ASD[:,i],
                                      alternative='two-sided')
        u1 = u1.reshape(n_freqs,n_times)
        u2 = n_TD*n_ASD - u1
        u = u1-u2
        p = p.reshape(n_freqs,n_times)
        
        plt.imshow(u, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   aspect='auto', origin='lower', cmap='RdBu_r')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title('%s' % cond)
        plt.colorbar()
        plt.savefig('%s/figures/misc/induced_power/TDvsASD_%s_mannWhitneyU_%s_new.png' 
                    % (paths['cluster'], cond, roi), dpi=300)
        plt.close()
        
        # thresholding
        u_th = np.zeros(u.shape)
        u_th[np.where(p<0.05)] = u[np.where(p<0.05)]
        plt.imshow(u_th, extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    aspect='auto', origin='lower', cmap='RdBu_r')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title('%s' % cond)
        plt.colorbar()
        plt.savefig('%s/figures/misc/induced_power/TDvsASD_%s_mannWhitneyUth05_%s_new.png' 
                    % (paths['cluster'], cond, roi), dpi=300)
        plt.close()