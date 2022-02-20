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
# from surfer import Brain
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import pingouin as pg
import seaborn as sns
import math
from helper_functions import pooled_stdev, p_from_r
import os

paths = {'cluster': '/autofs/cluster/transcend/jussi/',
         'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

f = open('%s/p/subjects.p' % paths['out'], 'rb')
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
n_subs = len(sub_IDs)
n_ASD = len([i for i in sub_info['ASD'] if i=='Yes'])
n_TD = len([i for i in sub_info['ASD'] if i=='No'])
inds_ASD = [ind for ind in range(n_subs) if sub_info['ASD'][ind]=='Yes']
inds_TD = [ind for ind in range(n_subs) if sub_info['ASD'][ind]=='No']
n_subs = n_ASD + n_TD

conds = ['MSS', 'SWS']
n_conds = len(conds)

regress_covars = False
covar_labels = ['age','NVIQ','VIQ']

# Time specifications
tmin = 0
tstep = 0.001
tmax = 1.5
times = np.arange(tmin,tmax,tstep)
n_times = int(len(times))

# Parameters for source estimate
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'MNE'
mode = 'mean_flip'
pick_ori = 'normal'

# Read the source space we are morphing to
src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
src = mne.read_source_spaces(src_fname)
fsave_verts = [s['vertno'] for s in src]
n_verts_fsave = len(fsave_verts[0])+len(fsave_verts[1])


#%% ROIs

hemis = ['lh', 'rh']
n_hemis = len(hemis)
roi_names = ['AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto']
n_rois = len(roi_names)

#%% Read timecourses from ROIs

erfs = np.zeros((n_subs, n_conds, n_hemis, n_rois, n_times))

for i_sub,sub_ID in enumerate(sub_info['sub_ID']):
    print(sub_ID)
    sub_path = '%s/%s/' % (paths['cluster'], sub_ID)
    inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-30Hz-oct6-inv.fif' 
                                                 % (sub_path, sub_ID))
    src = inv['src']    
    verts = [src[i]['vertno'] for i in range(2)]
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    epochs = mne.read_epochs('%s/%s_speech_1-30Hz_-200-2000ms-epo.fif' 
                             % (sub_path, sub_ID), proj=False)
    epochs = epochs[conds].equalize_event_counts(conds)[0]
    picks = mne.pick_types(epochs.info, meg=True)
    
    for i_hemi,hemi in enumerate(hemis):
        rois = []
        for roi_name in roi_names:
            roi_path = '%s/%s/rois/%s-%s.label' % (paths['cluster'], sub_ID, 
                                                   roi_name, hemi)
            if os.path.exists(roi_path):
                rois.append(mne.read_label(roi_path))
            else:
                rois.append(mne.read_labels_from_annot(fs_id, parc='PALS_B12_Lobes', 
                                                       subjects_dir=paths['fs'], 
                                                       regexp=roi_name)[0])
        
        for i_cond,cond in enumerate(conds):                                            
            evoked = epochs[cond].apply_baseline().average(picks=picks)
            stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method, 
                                                 pick_ori=pick_ori)
            stc.crop(tmin=tmin, tmax=tmax, include_tmax=False)
    
            for i_roi,roi in enumerate(rois):            
                # extract time course from ROIs
                erfs[i_sub,i_cond,i_hemi,i_roi] = stc.extract_label_time_course(roi, src, 
                                                                         mode=mode)[0]
    
erfs_abs = abs(erfs)

#%% Regress covariates out from the data

if regress_covars:
    print('Regressing out covariates: %s' % covar_labels)
    covars = []
    for covar_label in covar_labels:
        covars.append(sub_info[covar_label])
    model = np.zeros((n_subs,len(covars)+1))
    model[:,0] = np.ones(n_subs)
    model[:,1::] = np.transpose(covars)
       
    erfs = np.reshape(erfs_abs, (n_subs, n_conds*n_rois*n_times))
    beta = scipy.linalg.lstsq(model, erfs)[0]
    erfs = erfs - model.dot(beta)
    erfs = np.reshape(erfs, (n_subs, n_conds, n_rois, n_times))
else: 
    print('No covariates')
    covar_labels = ['noCovars']
    erfs = erfs_abs 


#%% Plot timecourses 

tmin = 0
tmax = 1.5

linewidth = 2 # for plots

for i_hemi,hemi in enumerate(hemis):
    for i_roi,roi_name in enumerate(roi_names):
        fig = plt.figure()
        plt.get_current_fig_manager().full_screen_toggle()
        ax = plt.gca()
        legend = []
        first_peak_latency = 200
        last_peak_latency = 500
        for i_cond,cond in enumerate(conds):
            if 'Speech' in cond: 
                linestyle = '-'
                color_TD = 'lightgreen'
                color_ASD = 'orchid'
            elif 'Jabber' in cond: 
                linestyle='-'
                color_TD = 'darkgreen'
                color_ASD = 'purple'
            elif 'Noise' in cond: 
                linestyle=':'
                color_TD = 'darkgreen'
                color_ASD = 'purple'
            elif 'SWS' in cond: 
                linestyle='-'
                color_TD = 'darkgreen'
                color_ASD = 'purple'
            elif 'MSS' in cond: 
                linestyle='-'
                color_TD = 'lightgreen'
                color_ASD = 'orchid'
                
            erf_TD = erfs[inds_TD,i_cond,i_hemi,i_roi].mean(0)
            temp = np.argmax(erf_TD[0:200])
            if temp<first_peak_latency:
                first_peak_latency = temp
            temp = np.argmax(erf_TD[500:800]) + 500
            if temp>last_peak_latency:
                last_peak_latency = temp
            plt.plot(erf_TD, label='TD %s' % cond, linewidth=linewidth, 
                     linestyle=linestyle, color=color_TD)
            legend.append('TD - %s' % cond)
            
            erf_ASD = erfs[inds_ASD,i_cond,i_hemi,i_roi].mean(0)
            temp = np.argmax(erf_ASD[0:200])
            if temp<first_peak_latency:
                first_peak_latency = temp
            temp = np.argmax(erf_ASD[500:800]) + 500
            if temp>last_peak_latency:
                last_peak_latency = temp
            plt.plot(erf_ASD, label='ASD %s' % cond, linewidth=linewidth, 
                     linestyle=linestyle, color=color_ASD)
            legend.append('ASD - %s' % cond)
            
        #    ax.set_xticklabels([str(i) for i in np.arange(int(tmin_cluster*1000)-100,int(tmax_cluster*1000)+1,100)])
            # vertical lines to indicate the time extent of the cluster
        #    plt.Axes.axvline(ax, x=int((tmin_clusters[roi_index]-tmin_cluster)*1000), 
        #                     color='k', linestyle='dashed', linewidth=linewidth)
        #    plt.Axes.axvline(ax, x=int((tmax_clusters[roi_index]-tmin_cluster)*1000), 
        #                     color='k', linestyle='dashed', linewidth=linewidth)
        #    plt.Axes.axvline(ax, x=int(peaks[roi_index]*1000), 
        #                     color='red', linestyle='solid', linewidth=2)
        #    ax.set_ylim(0.5*10**-11, 4.2*10**-11)
        #    plt.legend(loc='lower right')
            
        # plt.title(roi_name)
        ax.set_xticks(np.arange(tmin*1e3, tmax*1e3+1, 500))
        ax.set_xlim(0,1500)
        ax.set_yticks(np.arange(0, 7e-11, 1e-11))
        plt.xlabel('Time (ms)')
        plt.ylabel('Current strength (Am)')
        plt.Axes.axvline(ax, x=first_peak_latency, color='k', linestyle='dashed')
        plt.Axes.axvline(ax, x=last_peak_latency, color='k', linestyle='dashed')
        # plt.legend(legend, prop={'size': 6})
        fig.savefig('%s/figures/erf/tcs_%s_%dTD_%dASD_%s-%s.png' % (paths['cluster'], 
                                                                ('_').join(conds), 
                                                                n_TD, n_ASD, 
                                                                roi_name, hemi), dpi=300)
        plt.close()


#%% Contrasts
        
data_contrasts = []
contrasts = []
for i_cond,cond1 in enumerate(conds[0:-1]):
    for j_cond,cond2 in enumerate(conds[i_cond+1::], i_cond+1):
        data_contrasts.append(erfs[:, i_cond] - erfs[:, j_cond])
        contrasts.append('%s-%s' % (cond1, cond2))
        
dims = np.arange(len(np.shape(data_contrasts)))
dims_transposed = [dims[1]]+[dims[0]]+list(dims[2::]) # swap first two
data_contrasts = np.array(data_contrasts).transpose(dims_transposed)

# add contrasts data
erfs = np.concatenate((erfs, data_contrasts), axis=1)
del data_contrasts
conds = conds + contrasts
n_conds = len(conds)

#%% Mean plot
    
tmin = 0
tmax = 1500
mode = 'peak-to-peak'
    
cois = ['SWS']

group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
# group_labels = np.repeat(group_labels, n_conds)
# cond_labels = conds * n_subs

c_TD = 'lightgreen'
c_ASD = 'orchid'
c2_TD = 'darkgreen'
c2_ASD = 'purple'
    
for cond in cois:
    i_cond = conds.index(cond)
    for i_roi,roi_name in enumerate(roi_names):
        for i_hemi,hemi in enumerate(hemis):    
            if mode=='mean':
                this_data = erfs[:, i_cond, i_hemi, i_roi, tmin:tmax].mean(-1)
            elif mode=='peak':
                if cond=='MSS-SWS':
                    this_data = erfs[:, 0, i_hemi, i_roi, tmin:tmax].max(-1) -\
                        erfs[:, 1, i_hemi, i_roi, tmin:tmax].max(-1)
                else:
                    this_data = erfs[:, i_cond, i_hemi, i_roi, tmin:tmax].max(-1)
            elif mode=='peak-to-peak':
                this_data = erfs[:, i_cond, i_hemi, i_roi,
                                 first_peak_latency:last_peak_latency].mean(-1)
                
            data_pd = {'ERFs': this_data, 
                       'Group': group_labels,
                       # 'Condition': cond_labels,
                       'ID': sub_IDs}
            
            data_frame = pd.DataFrame(data=data_pd)
            
            # plt.figure()
            # sns.barplot(x='Condition', y='ERFs', hue='Group', data=data_frame,
            #               alpha=1, ci=68, edgecolor='gray', palette=[c_TD, c_ASD],
            #               hue_order=['TD', 'ASD'])
            
            # if cond=='SWS':
            #     c_TD = c2_TD
            #     c_ASD = c2_ASD
            fig, ax = plt.subplots()
            ax = sns.violinplot(x='Group', y='ERFs', 
                             data=data_frame, alpha=1, edgecolor='gray', 
                             palette=[c_TD, c_ASD], order=['TD', 'ASD'],
                             inner=None, cut=0)
            # sns.swarmplot(x='Group', y='Connectivity (%s)' % con_method, 
            #                 data=data_frame, size=10, alpha=1, 
            #                 edgecolor='black', linewidth=1,
            #                 palette=[c_TD, c_ASD], order=['TD', 'ASD'])
            sns.stripplot(x='Group', y='ERFs', 
                            data=data_frame, size=10, jitter=0.05, 
                            alpha=1, edgecolor='black', linewidth=1,
                            palette=[c_TD, c_ASD], order=['TD', 'ASD'])
            
            # figure size
            # y_max = np.max(this_data)
            # y_min = np.min(this_data)
            y_max = 3.3e-11
            y_min = -3.3e-11
            if '-' in cond:
                ax.set_ylim(y_min * 1.1, y_max * 1.1)
            else:
                ax.set_ylim(0, 1.4e-10)
            fig.set_size_inches(6,12)  
            
            # draw mean & sem
            for i, inds_group in enumerate([inds_TD, inds_ASD]):
                mean_line_len = 0.12
                mean_line_wid = 6
                sem_line_len = mean_line_len * 0.5
                sem_line_wid = mean_line_wid * 0.5
                data_group = this_data[inds_group]
                x = (i-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]) # scaling
                ymin = np.mean(data_group)-scipy.stats.sem(data_group, ddof=0)
                ymax = np.mean(data_group)+scipy.stats.sem(data_group, ddof=0)
                yminScaled = (ymin-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
                ymaxScaled = (ymax-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
                ax.axhline(np.mean(data_group), xmin=x-mean_line_len, xmax=x+mean_line_len, 
                           color='k', lw=mean_line_wid, solid_capstyle='round', zorder=100)
                ax.axhline(np.mean(data_group)+scipy.stats.sem(data_group), 
                           xmin=x-sem_line_len, xmax=x+sem_line_len, color='k', 
                           lw=sem_line_wid, solid_capstyle='round', zorder=100)
                ax.axhline(np.mean(data_group)-scipy.stats.sem(data_group), 
                           xmin=x-sem_line_len, xmax=x+sem_line_len, color='k', 
                           lw=sem_line_wid, solid_capstyle='round', zorder=100)
                ax.axvline(i, yminScaled, ymaxScaled, color='k', lw=sem_line_wid)
                
            # plt.title('%s - %s' % (roi_name, hemi))
            # plt.rcParams['axes.labelsize'] = 14
            plt.savefig('%s/figures/erf/violin_%s_%s-%s_%s%d-%dms.png' % (paths['cluster'], cond, 
                                                             roi_name, hemi, mode, tmin, tmax), dpi=300)
            plt.close()

#%% Correlations
    
plot = False
partial_corr = False

tmin = 0
tmax = 1600
    
print('Correlation between ERFs and behavioral scores\n')
score_labels = ['age', 'NVIQ', 'VIQ', 'SRS_tot_T', 'ASPS', 'ICSS', 'SCSS'] #   
for i_roi,roi_name in enumerate(roi_names):
    for i_cond,cond in enumerate(conds):
        for score_label in score_labels:
            for inds_group,group in zip([inds_TD, inds_ASD], ['TD', 'ASD']):
                if group=='TD': 
                    color = 'lightgreen'
                else:
                    color = 'orchid'        
            
                brains = list(erfs[inds_group, i_cond, i_roi, tmin:tmax].mean(-1))
                scores = list(np.array(sub_info[score_label])[inds_group])
                # indices of None, i.e. missing scores
                inds_null = [i for i in range(len(scores)) if scores[i]==None]
                inds_null.reverse() # reverse so that largest are removed first
                # remove Nones and the corresponding brain data value
                for i in inds_null:
                    scores.pop(i)
                    brains.pop(i)
                    
                if partial_corr:
                    this_model = np.delete(model[inds_group], inds_null, axis=0)
                    beta = scipy.linalg.lstsq(this_model, scores)[0]
                    scores = scores - this_model.dot(beta)
                    
                if len(scores)>0:
                    r, p = scipy.stats.pearsonr(brains, scores)
                    if p<0.05 and group=='ASD':
                        print('%s' % roi_name)
                        print('%s - %s - %s: r=%.2f, p=%.3f' % (group, cond, 
                                                                score_label, r, p))
                # else:
                #     print('No %s scores for %s' % (score_label, group))
                    
                if plot:
                    sns.regplot(scores, brains, ci=None, scatter_kws={'s':80}, 
                                color=color, marker='o')
            if plot:
                plt.xlabel(score_label)
                plt.ylabel('Entrainment')
                plt.savefig('%s/figures/erf/correlation_%s_%s_%s_%s-%s.png' % (paths['cluster'], 
                                                                               cond, score_label,  
                                                                               roi_name), dpi=300)
                plt.close()


#%% Laterality index
    
tmin = 0
tmax = 1600
    
# values have to positive
min_val = np.min(erfs)
if min_val < 0:
    erfs = erfs + abs(min_val)

LI = []
for i_roi in range(n_rois):
    LI.append((erfs[:, :, 1, i_roi, tmin:tmax].mean(-1) - 
               erfs[:, :, 0, i_roi, tmin:tmax].mean(-1)) / 
              (erfs[:, :, 1, i_roi, tmin:tmax].mean(-1) + 
               erfs[:, :, 0, i_roi, tmin:tmax].mean(-1)))
    
LI = np.array(LI)

#%% Plot LI

group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
group_labels = np.repeat(group_labels, n_conds)
cond_labels = conds * n_subs

c_TD = 'lightgreen'
c_ASD = 'orchid'

for i_roi,roi in enumerate(rois):
    data_pd = {'LI': np.concatenate(LI[i_roi]), 
               'Group': group_labels,
               'Condition': cond_labels,
               'ID': np.repeat(sub_info['sub_ID'], n_conds)}
    
    data_frame = pd.DataFrame(data=data_pd)
    
    # sns.stripplot(x='Condition', y='LI', hue='Group', data=data_frame,
    #               alpha=1, edgecolor='gray', palette=[c_ASD,c_TD])
    sns.barplot(x='LI', y='Condition', hue='Group', data=data_frame, 
                hue_order=['TD', 'ASD'], ci=68, edgecolor='gray', 
                palette=[c_TD, c_ASD], estimator=np.mean, orient='h')
    plt.rcParams['axes.labelsize'] = 12
    plt.savefig('%s/figures/erf/LI_%s.png' % (paths['cluster'], 
                                              roi.name[0:-3]), dpi=300)
    plt.close()
