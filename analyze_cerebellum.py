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
         'local': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cb': '/autofs/cluster/transcend/data_exchange/cerebellar_source_spaces_autism_cohort/source_spaces'
         }

f = open('%s/p/subjects.p' % paths['local'], 'rb')
sub_info = pickle.load(f)
f.close()

# define bad subjects
bad_subs = ['100001', '105801', '107301']
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

conds = ['Speech', 'Jabber', 'MSS', 'SWS', 'Noise']
n_conds = len(conds)
equalize_epoch_counts = True

regress_covars = False
covar_labels = ['age','NVIQ','VIQ']

# Time specifications
tmin = 0
tmax = 2.0
tstep = 1e-3
n_times = len(np.arange(tmin, tmax, tstep))

# Parameters for source estimate
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'sLORETA'

#%% Read timecourses from ROIs

gfp_cx = np.zeros((n_subs, n_conds, n_times))
gfp_cb = np.zeros((n_subs, n_conds, n_times))

for i_sub,sub_ID in enumerate(sub_info['sub_ID']):
    print(sub_ID)
    path_local = '%s/%s/' % (paths['local'], sub_ID)
    path_cluster = '%s/%s/' % (paths['cluster'], sub_ID)
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    inv = mne.minimum_norm.read_inverse_operator('%s/%s_speech_1-30Hz-cb-inv.fif' 
                                                 % (path_local, sub_ID))
    src = inv['src']
    verts_cx = src[0]['vertno'] # cortex vertices
    verts_cb = src[1]['vertno'] # cerebellum vertices
    
    epochs = mne.read_epochs('%s/%s_speech_1-30Hz_-200-2000ms-epo.fif' 
                             % (path_cluster, sub_ID), proj=False)
    picks = mne.pick_types(epochs.info, meg=True)
    
    if equalize_epoch_counts:
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
    n_epochs = len(epochs[conds[0]])
        
    for i_cond, cond in enumerate(conds):                                            
        evoked = epochs[cond].apply_baseline().average(picks=picks)
        stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method, 
                                             pick_ori='normal')
        stc.crop(tmin=tmin, tmax=tmax, include_tmax=False)
        gfp_cx[i_sub, i_cond] = stc.data[0:len(verts_cx)].std(0)
        gfp_cb[i_sub, i_cond] = stc.data[len(verts_cx)::].std(0)

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

cerebellum = False # cerebellum or cortex
cois = ['MSS', 'SWS']# 'Speech', 'Jabber', 'Noise'

if cerebellum:
    this_data = gfp_cb
else:
    this_data = gfp_cx
    
tmin = 0
tmax = 2.0

linewidth = 2 # for plots

fig = plt.figure()
plt.get_current_fig_manager().full_screen_toggle()
ax = plt.gca()
legend = []
for i_cond,cond in enumerate(conds):
    if cond not in cois:
        continue
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
    
    plt.plot(this_data[inds_TD,i_cond].mean(0), label='TD %s' % cond, 
             linewidth=linewidth, linestyle=linestyle, color=color_TD)
    legend.append('TD - %s' % cond)
    plt.plot(this_data[inds_ASD,i_cond].mean(0), label='ASD %s' % cond, 
             linewidth=linewidth, linestyle=linestyle, color=color_ASD)
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
    
plt.xlabel('Time (ms)')
plt.ylabel('sLORETA') # Current strength (Am)
plt.legend(legend, prop={'size': 6})
if cerebellum:
    plt.title('Cerebellum - global field power')
    fig.savefig('%s/figures/erf/cb_gfp_%s_%dASD_%dTD_%s.png' % (paths['cluster'], 
                                                        ('_').join(cois), 
                                                        n_ASD, n_TD, method), dpi=300)
else:
     plt.title('Cortex - global field power')
     fig.savefig('%s/figures/erf/cx_gfp_%s_%dASD_%dTD_%s.png' % (paths['cluster'], 
                                                        ('_').join(cois), 
                                                        n_ASD, n_TD, method), dpi=300)

plt.close()


#%% Bar graphs
    
this_data = gfp_cb
cois = ['MSS', 'SWS']
tmin = 200
tmax = 500

inds_cois = [conds.index(coi) for coi in cois]
    
group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
group_labels = np.repeat(group_labels, len(cois))
cond_labels = cois * n_subs

c_TD = 'lightgreen'
c_ASD = 'orchid'   

data_pd = {'GFP': np.concatenate(this_data[:,inds_cois,tmin:tmax].mean(-1)), 
           'Group': group_labels,
           'Condition': cond_labels,
           'ID': np.repeat(sub_IDs, len(cois))}

data_frame = pd.DataFrame(data=data_pd)

plt.figure()
sns.barplot(x='Condition', y='GFP', hue='Group', data=data_frame,
              alpha=1, ci=68, edgecolor='gray', palette=[c_TD, c_ASD],
              hue_order=['TD', 'ASD'])
# sns.stripplot(x='Condition', y='Entrainment', hue='Group', data=data_frame,
#               alpha=1, edgecolor='gray', palette=[c_ASD,c_TD])
# sns.violinplot(x='Condition', y='Entrainment', hue='Group', data=data_frame,
#               cut=0, inner='point', alpha=1, ci=68, edgecolor='gray', 
#               palette=[c_ASD,c_TD])
plt.title('Cerebellum - global field power (%d-%dms)' % (tmin, tmax))
plt.rcParams['axes.labelsize'] = 14
plt.savefig('%s/figures/erf/bars_cb_gfp_%d-%dms_%s.png' % (paths['cluster'], 
                                                           tmin, tmax, 
                                                           ('_').join(cois)), dpi=300)
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
