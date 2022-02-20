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
from helper_functions import compare_corr_coefs
import pandas as pd
import seaborn as sns
import pingouin as pg
from matplotlib import pyplot as plt
import sys
import random
import os

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

fname = 'pac_vertexwise_seed_AC_wpli2_debiased_8-12Hz_28TD_vs_28ASD_MSS-SWS_cdt0.001_phtNone-lh_MSS-SWS-Baseline_8-12Hz_30-60Hz_0-1500ms_ozkurt_concatEpochs_seedAC.p'

# freqs of interest
foi_phase = [8, 12]
foi_amp = [30, 60]

covar_labels = [] # regress 

#% load one file to get variables
f = open('%s/%s/p/%s' % (paths['cluster'], sub_info['sub_ID'][0], fname), 'rb')
data = pickle.load(f)
f.close()

conds_orig = list(data['pac'].keys())
cond_labels = [cond.replace('/','') for cond in conds_orig]
n_conds_orig = len(conds_orig)

n_verts = data['n_spats']
rois = data['rois']
pac_method = data['pac_method']

freqs_phase = data['phase_fq_range']
phase_fq_min = freqs_phase[0]
phase_fq_max = freqs_phase[-1]
if phase_fq_min != phase_fq_max:
    phase_fq_step = freqs_phase[1]-freqs_phase[0]
n_freqs_phase = len(freqs_phase)

freqs_amp = data['amp_fq_range']
amp_fq_min = freqs_amp[0]
amp_fq_max = freqs_amp[-1]
if amp_fq_min != amp_fq_max:
    amp_fq_step = freqs_amp[1]-freqs_amp[0]
n_freqs_amp = len(freqs_amp)

# read the source space we are morphing to
src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
src_fsave = mne.read_source_spaces(src_fname)
adjacency = mne.spatial_src_adjacency(src_fsave)
verts_fsave = [s['vertno'] for s in src_fsave]
n_verts_fsave = len(np.concatenate(verts_fsave))


#% load data

i_fmin_phase = list(freqs_phase).index(foi_phase[0])
i_fmax_phase = list(freqs_phase).index(foi_phase[1])
i_fmin_amp = list(freqs_amp).index(foi_amp[0])
i_fmax_amp = list(freqs_amp).index(foi_amp[1])

data = np.zeros((n_subs, n_conds_orig, n_verts_fsave))
for i_sub, sub_ID in enumerate(sub_info['sub_ID']):
    print(sub_ID)
    sub_path = '%s/%s/' % (paths['cluster'], sub_ID)
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                                 % (sub_path, sub_ID))     
    src = inv['src']
    verts = [s['vertno'] for s in src]
    n_verts = len(np.concatenate(verts))
    
    f = open('%s/p/%s' % (sub_path, fname), 'rb')
    data_sub = pickle.load(f)
    f.close()
    
    for i_cond, cond in enumerate(conds_orig):
        stc_data = np.zeros(n_verts)        
            
        # get the mask that was used to calculate PAC
        mask = data_sub['mask']
        # get vertices of the mask
        if mask.hemi=='lh':
            verts_hemi = verts[0]
            verts_mask = mask.get_vertices_used(vertices=verts_hemi)
        else:
            verts_hemi = verts[1]
            verts_mask = mask.get_vertices_used(vertices=verts_hemi)
        inds_mask = np.searchsorted(verts_hemi, verts_mask)
        
        # put data in stc array
        data_cond = np.array(data_sub['pac'][cond])[:, i_fmin_phase:i_fmax_phase, 
                                                    i_fmin_amp:i_fmax_amp].squeeze()
        # data_cond = np.array(data_sub['pac'][cond])[:, i_fmin_phase:i_fmax_phase].squeeze()
        # if not foi_amp and not foi_phase:
        #     data_cond = np.array(data_sub['pac'][cond]).squeeze()
            
        if len(data_cond.shape)>1:
            data_cond_mean = data_cond.mean(tuple(np.arange(1, len(data_cond.shape))))
        stc_data[inds_mask] = data_cond_mean
        
        # fill the array with small random numbers
        inds_zero = list(set(inds_mask) ^ set(np.arange(n_verts)))
        stc_data[inds_zero] = np.random.random(len(inds_zero)) / 1e6
        
        # make stc
        stc = mne.SourceEstimate(stc_data, vertices=verts, tmin=0, 
                                 tstep=0.001, subject=fs_id)

        # morph stc to fsaverage
        morph = mne.compute_source_morph(stc, fs_id, 'fsaverageJA', spacing=verts_fsave, 
                                         subjects_dir=paths['fs'])
        stc = morph.apply(stc)
        
        # put morphed data to group data array
        data[i_sub, i_cond] = stc.data.squeeze()

#% Contrasts       
contrasts = []
data_contrasts = []
for i_cond,cond1 in enumerate(conds_orig[0:-1]):
    for j_cond,cond2 in enumerate(conds_orig[i_cond+1::], i_cond+1):
        data_contrasts.append(data[:, i_cond] - data[:, j_cond])
        contrasts.append('%s-%s' % (cond1, cond2))
data_contrasts = np.array(data_contrasts).transpose(1,0,2)

# Add contrasts data
data_orig = np.concatenate((data, data_contrasts), axis=1)
conds = conds_orig + contrasts
n_conds = len(conds)

#% Regress covariates    
if covar_labels:
    print('Regressing out covariates: %s' % covar_labels)
    covar_out = '%sReg' % ('').join(covar_labels)
    covars = []
    for covar_label in covar_labels:
        covars.append(sub_info[covar_label])
    model = np.zeros((n_subs, len(covars)+1))
    model[:,0] = np.ones(n_subs)
    model[:,1::] = np.transpose(covars)
       
    # data = np.transpose(data_orig, (1,0,2,3,4))
    data = np.reshape(data, (n_subs, n_conds*n_verts_fsave))
    beta = scipy.linalg.lstsq(model, data)[0]
    data = data - model.dot(beta)
    # intercept = beta[0]
    # data += intercept
    data = np.reshape(data, (n_subs, n_conds, n_verts_fsave))
    # data = np.transpose(data, (1,0,2,3,4))
else: 
    # data = data_orig
    print('No covariates')
    covar_out = 'noReg'

#% Exclude regions outside the mask
path_mask = '%s/rois/%s.label' % (paths['cluster'], mask.name)
if os.path.exists(path_mask):
    mask_fsave = mne.read_label('%s' % path_mask)
else:
    mask_fsave = mne.read_labels_from_annot('fsaverageJA', parc='PALS_B12_Lobes', 
                                            subjects_dir=paths['fs'], regexp=mask.name)[0]

if mask_fsave.hemi=='lh':
    verts_hemi_fsave = verts_fsave[0]
    verts_mask_fsave = mask_fsave.get_vertices_used(vertices=verts_hemi_fsave)
else:
    verts_hemi_fsave = verts_fsave[1]
    verts_mask_fsave = mask_fsave.get_vertices_used(vertices=verts_hemi_fsave)

inds_mask_fsave = np.searchsorted(verts_hemi_fsave, verts_mask_fsave)
inds_exclude = list(set(inds_mask_fsave) ^ set(np.arange(n_verts_fsave)))

exclude = np.zeros(n_verts_fsave, dtype=bool)
exclude[inds_exclude] = True


#%% Test for difference between groups
cois = ['MSS-SWS'] # conditions of interest

# th = dict(start=0, step=1)
cdt_p = 0.01
th = scipy.stats.distributions.f.ppf(1. - cdt_p / 2., n_ASD - 1, n_TD - 1)
n_perms = 5000
hemi = 'lh'
plot_cluster = False
save_roi = False

do_correlations = True
save_brains = False
corr_type = 'partial' # 'basic', 'partial' ,'semipartial'
corr_method = 'pearson'
covar_labels = ['age'] # for partial correlation
ci = 95 # confidence interval
corr_conds = ['SWS-Baseline', 'MSS-Baseline', 'MSS-SWS']
score_labels = ['VIQ', 'ICSS', 'SCSS', 'ASPSa1-a5', 'SRS_tot_T', 
                'SRS_socComm_T', 'ADOS_tot_old', 'ADOS_comm_soc_old'] 
pick_val = 'peak' # 'peak' or 'mean'
plot_correlation = True
loc_legend = 'right'

mean_plot = False #'violin'
nonparametric = True # two-sample t-test of nonparametric wilcoxon rank sums
plot_conds = ['MSS-SWS']
mark_signifs = False

freqs_phase_label = '%.1f-%.1fHz' % (freqs_phase[i_fmin_phase], 
                                     freqs_phase[i_fmax_phase])
freqs_amp_label = '%d-%dHz' % (freqs_amp[i_fmin_amp], 
                               freqs_amp[i_fmax_amp])
out_id = 'phase%s_amp%s' % (freqs_phase_label, freqs_amp_label)

for i_cond, cond in enumerate(conds):
    if cond not in cois: continue
    print('\n%s' % cond)            
    stat_vals, clusters, cluster_pvals, H0 = \
        mne.stats.permutation_cluster_test([data[inds_TD, i_cond], 
                                            data[inds_ASD, i_cond]], 
                                           adjacency=adjacency, 
                                           n_permutations=n_perms+1,
                                           n_jobs=12, threshold=th, 
                                           out_type='mask', seed=8,
                                           exclude=exclude)
    #  Select significant (multiple-comparisons corrected) clusters
    good_clust_inds = np.where(cluster_pvals < 0.1)[0]
    cluster_mask = np.zeros(n_verts_fsave, dtype=bool)
    print('Min p-val: %f' % np.min(cluster_pvals))    
    if good_clust_inds.size > 0:
        stat_vals_th = np.zeros(n_verts_fsave)
        if th=='TFCE':
            stat_vals_th[good_clust_inds] = stat_vals[good_clust_inds]
            # plot_max_val = np.max(stat_vals_th)
        else:
            for good_clust_ind in range(len(good_clust_inds)):
                stat_vals_th[clusters[good_clust_ind]] = \
                stat_vals[clusters[good_clust_ind]]
            
        if plot_cluster:
            stc = mne.SourceEstimate(stat_vals_th, vertices=verts_fsave, 
                                      tmin=0, tstep=0.001, subject='fsaverageJA')
            brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'], 
                            hemi=hemi, backend='matplotlib', spacing='ico7',
                            background='k', colorbar=True,
                            clim=dict(kind='value', lims=[0.0001, 0.0001, 10]))
            plt.title('F-value', color='w')
            brain.text(0.5, 0.8, 'p=%.3f' % np.min(cluster_pvals), 
                        horizontalalignment='center', fontsize=14, color='w')
            brain.savefig('%s/figures/pac/TD_vs_ASD_%s_%s-%s_cdt%s_%s.png' % (paths['cluster'], 
                                                                              cond, out_id, hemi, 
                                                                              cdt_p, covar_out), dpi=500)
            plt.close()
        
        cluster_mask[np.nonzero(stat_vals_th)] = True   
        
        if save_roi:
            labels = mne.stc_to_label(stc, src_fsave, smooth=True, 
                                      connected=True, subjects_dir=paths['fs'])[0]
            n_verts = 0
            for i,this_label in enumerate(labels):
                temp = len(this_label.get_vertices_used())
                if temp > n_verts: 
                    label = labels[i]
                    n_verts = temp
                    
            label.save('%s/rois/pac_%s_%dTD_vs_%dASD_%s_cdt%s-%s.label' 
                       % (paths['cluster'], out_id, n_TD, n_ASD, cond, cdt_p, hemi))
    
    if do_correlations and np.any(cluster_mask):
        if corr_type=='partial':
            print('Partial correlation, %s as covariates' % covar_labels)
        elif corr_type=='semipartial':
            print('Semipartial correlation, %s as covariates' % covar_labels)
        else:
            print('Basic correlation')
        for i_cond,cond in enumerate(conds):
            if cond not in corr_conds:
                continue
            for score_label in score_labels:
                if pick_val=='peak':
                    brains = data[:, i_cond, cluster_mask].max(-1)
                elif pick_val=='mean':
                    brains = data[:, i_cond, cluster_mask].mean(-1)
                    
                if save_brains:
                    np.save('%s/npy/pac_%s_N%d_%s_cdt%s_%s_%s' % (paths['cluster'], 
                                                                out_id, n_subs, cond, 
                                                                cdt_p, covar_out, 
                                                                pick_val), brains)
                    
                scores = np.array(sub_info[score_label])
                diagnosed = sub_info['ASD'].copy()
                this_sub_IDs = sub_info['sub_ID'].copy()
                
                # indices of None, i.e. missing scores
                null_inds = [i for i in range(len(scores)) 
                             if scores[i]==None]
                null_inds.reverse() # reverse so that largest index is removed first
                # remove Nones and the corresponding brain data
                for i in null_inds:
                    scores = np.delete(scores, i)
                    brains = np.delete(brains, i, axis=0)
                    diagnosed.pop(i)   
                    this_sub_IDs.pop(i)
                    
                this_inds_TD = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
                this_inds_ASD = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
                    
                # if partial and covar_labels:
                #     #% Regress covariates from scores
                #     covars = []
                #     for covar_label in covar_labels:
                #         covars.append(sub_info[covar_label])
                #     for i in null_inds: 
                #         covars = np.delete(covars, i, axis=1)
                #     model = np.zeros((len(this_sub_IDs),len(covars)+1))
                #     model[:,0] = np.ones(len(this_sub_IDs))
                #     model[:,1::] = np.transpose(covars)
                       
                #     beta = scipy.linalg.lstsq(model, list(scores))[0]
                #     scores = list(scores) - model.dot(beta)
                    
                #     # zscore scores
                #     scores = scipy.stats.zscore(scores, axis=None)
                # if covar_labels:
                #     brains = scipy.stats.zscore(brains, axis=None)
                    
                       
                fn_out = '%s/figures/pac/corr_cluster_%s_%s_%s_%s_cdt%s_%s' % (paths['cluster'],  
                                                                            out_id, cond, score_label,
                                                                            pick_val, cdt_p, covar_out)
                p_TD = 1
                if len(scores[this_inds_TD])>0:
                    if corr_type=='basic':
                        r_TD, p_TD = pearsonr(scores[this_inds_TD], brains[this_inds_TD])
                    else:
                        data_pd = {'brains': list(brains[this_inds_TD]), 
                                   'scores': list(scores[this_inds_TD])}
                        for covar_label in covar_labels:
                            covar = np.array(sub_info[covar_label])[this_inds_TD]
                            data_pd.update({covar_label: list(covar)})
                        data_frame = pd.DataFrame(data=data_pd)
                        
                        if corr_type=='partial':
                            stats = pg.partial_corr(data=data_frame, x='brains', 
                                                    y='scores', covar=covar_labels,
                                                    method=corr_method)
                        elif corr_type=='semipartial':
                            stats = pg.partial_corr(data=data_frame, x='brains', 
                                                    y='scores', x_covar=covar_labels,
                                                    method=corr_method)
                        r_TD = stats['r'][0]
                        p_TD = stats['p-val'][0]
                        
                    if p_TD<0.1:
                        print('\nTD: %s vs. %s: r=%f, p=%f' 
                              % (cond, score_label, r_TD, p_TD))                    

                p_ASD = 1
                if len(scores[this_inds_ASD])>0:
                    if corr_type=='basic':
                        r_ASD, p_ASD = pearsonr(scores[this_inds_ASD], brains[this_inds_ASD])
                    else:
                        data_pd = {'brains': list(brains[this_inds_ASD]), 
                                   'scores': list(scores[this_inds_ASD])}
                        for covar_label in covar_labels:
                            covar = np.array(sub_info[covar_label])[this_inds_ASD]
                            data_pd.update({covar_label: list(covar)})
                        data_frame = pd.DataFrame(data=data_pd)
                        
                        if corr_type=='partial':
                            stats = pg.partial_corr(data=data_frame, x='brains', 
                                                    y='scores', covar=covar_labels)
                        elif corr_type=='semipartial':
                            stats = pg.partial_corr(data=data_frame, x='brains', 
                                                    y='scores', x_covar=covar_labels)
                        r_ASD = stats['r'][0]
                        p_ASD = stats['p-val'][0]

                    if p_ASD<0.1:
                        print('\nASD: %s vs. %s: r=%f, p=%f' 
                              % (cond, score_label, r_ASD, p_ASD))
                    
                p = 1
                if len(scores[this_inds_TD])>0 and len(scores[this_inds_ASD])>0:
                    z, p = compare_corr_coefs(r_TD, r_ASD, len(this_inds_TD), len(this_inds_ASD))
                    if p<0.1:
                        print('\nTD vs. ASD: %s vs. %s: z=%f, p=%f'
                              % (cond, score_label, z, p))
                    
                if plot_correlation and (p_ASD<0.05 or p_TD<0.05 or p<0.05):
                    if loc_legend == 'left':
                        loc = 0.025
                    elif loc_legend == 'right':
                        loc = 0.7
                    plt.figure()
                    if len(scores[this_inds_TD])>0:
                        ax = sns.regplot(list(scores[this_inds_TD]), brains[this_inds_TD], ci=ci, 
                                    scatter_kws={'s':80}, color='lightgreen', marker='o', label='TD')
                        if p_TD < 0.001:
                            p_TD =  eval('%.0e' % p_TD)
                        else:
                            p_TD = format(p_TD, '.3f')
                        ax.text(loc, 0.95, 'r=%.2f, p=%s' % (r_TD, p_TD), fontsize=12,                                    
                                color='lightgreen', transform=ax.transAxes)
                    if len(scores[this_inds_ASD])>0:
                        ax = sns.regplot(list(scores[this_inds_ASD]), brains[this_inds_ASD], ci=ci, 
                                    scatter_kws={'s':80}, color='orchid', marker='o', label='ASD')
                        if p_ASD < 0.001:
                            p_ASD =  eval('%.0e' % p_ASD)
                        else:
                            p_ASD = format(p_ASD, '.3f')
                        ax.text(loc, 0.9, 'r=%.2f, p=%s' % (r_ASD, p_ASD), fontsize=12,                                    
                                color='orchid', transform=ax.transAxes)     
                        
                    if len(scores[this_inds_TD])>0 and len(scores[this_inds_ASD])>0:
                        if p < 0.001:
                            p =  eval('%.0e' % p)
                        else:
                            p = format(p, '.3f')
                        ax.text(loc, 0.85, 'z=%.2f, p=%s' % (z, p), fontsize=12,                                    
                                color='k', transform=ax.transAxes) 
                    
                    ax.set_xlim(min(scores)-(max(scores)-min(scores))*0.05, 
                                max(scores)+(max(scores)-min(scores))*0.05)
                    ax.set_ylim(min(brains)-(max(brains)-min(brains))*0.05, 
                                max(brains)+(max(brains)-min(brains))*0.05)
                    plt.title('%s %s within cluster correlation' \
                              % (cond, out_id))
                    plt.xlabel('%s' % score_label)
                    plt.ylabel('%s' % cond) 
                    ax.legend(loc='lower left')
                    plt.tight_layout()
                    plt.savefig('%s.png' % fn_out, dpi=300)
                    plt.close()
                    
    if mean_plot and np.any(cluster_mask):
        group_labels = sub_info['ASD'] * len(plot_conds)
        # replace 'Yes' with 'ASD' and 'No' with 'TD' in group labels
        group_labels = ['ASD' if group_labels[i]=='Yes' else 'TD' 
                        for i in range(len(group_labels))]
        cond_labels = np.repeat(plot_conds, n_subs).tolist()
        
        inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
        inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
        
        if pick_val=='peak':
            this_data = {cond: data[:, conds.index(cond), cluster_mask].max(-1) for cond in plot_conds}
        elif pick_val=='mean':
            this_data = {cond: data[:, conds.index(cond), cluster_mask].mean(-1) for cond in plot_conds}
        data_pd = {'PAC': np.concatenate(list(this_data.values())), 
                   'Group': group_labels,
                   'Condition': cond_labels,
                   'ID': sub_info['sub_ID'] * len(plot_conds)}
        data_frame = pd.DataFrame(data=data_pd)
        
        c_TD = 'lightgreen'
        c_ASD = 'orchid'
        
        if mean_plot=='bar':
            plt.figure()
            ax = sns.barplot(x='Condition', y='PAC', hue='Group', 
                             data=data_frame, alpha=1, edgecolor='gray', ci=68, 
                             palette=[c_TD, c_ASD], hue_order=['TD', 'ASD'])
            plt.title('PAC cluster phase %d-%dHz, amp %d-%dHz' % (foi_phase[0], foi_phase[1],
                                                                  foi_amp[0], foi_amp[1]))
            plt.legend(loc='lower right')
            
        elif mean_plot=='violin':
            if len(plot_conds)==1:
                
                fig, ax = plt.subplots()
                ax = sns.violinplot(x='Group', y='PAC', data=data_frame, 
                                    alpha=1, edgecolor='gray', 
                                     palette=[c_TD, c_ASD], order=['TD', 'ASD'],
                                     inner=None, cut=0)
                # sns.swarmplot(x='Group', y='Connectivity (%s)' % con_method, 
                #                 data=data_frame, size=10, alpha=1, 
                #                 edgecolor='black', linewidth=1,
                #                 palette=[c_TD, c_ASD], order=['TD', 'ASD'])
                sns.stripplot(x='Group', y='PAC', data=data_frame, size=10, 
                              jitter=0.05, alpha=1, edgecolor='black', linewidth=1,
                              palette=[c_TD, c_ASD], order=['TD', 'ASD'])
                
                # figure size
                # y_max = np.max(this_data[plot_conds[0]])
                # y_min = np.min(this_data[plot_conds[0]])
                y_max = 0.008
                y_min = -0.008
                ax.set_ylim(y_min * 1.1, y_max * 1.1)
                fig.set_size_inches(6,12)  
                
                # draw mean & sem
                for i, inds_group in enumerate([inds_TD, inds_ASD]):
                    mean_line_len = 0.12
                    mean_line_wid = 6
                    sem_line_len = mean_line_len * 0.5
                    sem_line_wid = mean_line_wid * 0.5
                    data_group = this_data[plot_conds[0]][inds_group]
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
                    ax.axvline(i, yminScaled, ymaxScaled, color='k', lw=sem_line_wid,
                               zorder=100)
        
        stat, p, r = [], [], []
        for i_cond,cond in enumerate(plot_conds):
            if nonparametric:
                stats1 = ranksums(this_data[cond][inds_TD], this_data[cond][inds_ASD])
                stats2 = pg.mwu(this_data[cond][inds_TD], this_data[cond][inds_ASD])
            else:
                stats1 = ttest_ind(this_data[cond][inds_TD], this_data[cond][inds_ASD])
            stat.append(stats1[0])
            p.append(stats1[1])
            r.append(stats2['RBC'][0])
            
            print('%s: z=%.2f, p=%.5f, r=%.2f' % (cond, stat[i_cond], 
                                                  p[i_cond], r[i_cond]))
        if mark_signifs:
            signifs = []
            for val in p:
                if val>0.06: signifs.append('')
                elif val<0.06 and val>0.05: signifs.append('(*)')
                elif val<0.05 and val>0.01: signifs.append('*')
                elif val<0.01 and val>0.001: signifs.append('**')
                elif val<0.001: signifs.append('***')
                
            yloc = ax.get_ybound()[1] - (ax.get_ybound()[1] * 0.2)
            for signif, xloc in zip(signifs, ax.get_xticks()):
                ax.text(xloc, yloc, signif, fontsize=20, horizontalalignment='center') 
                
        plt.tight_layout()
        fn_out = '%s/figures/pac/%s_%dTD_%dASD_%s_%s_%s_cdt%s' \
                % (paths['cluster'], mean_plot, n_TD, n_ASD, out_id, cond, pick_val, cdt_p)
        if nonparametric:
            fn_out += '_ranksum'
        else:
            fn_out += '_ttest'
        fn_out += '_%s' % covar_out
        plt.savefig('%s.png' % fn_out, dpi=300)
        plt.close()
        
#%% vertexwise correlation

corr_conds = ['MSS', 'SWS']   
score_labels = ['age', 'ASPSa1-a5']#'age', 'ASPSa1-a5', 'ICSS', 'SRS_tot_T', 'ADOS_tot_old'] 
th_p = 0.05
partial = False
plot_correlation = True

if foi_phase and foi_phase[0]==foi_phase[-1]:
    freqs_phase_label = '%.1fHz' % foi_phase[0]
elif foi_phase and foi_phase[0]!=foi_phase[-1]:
    freqs_phase_label = '%.1f-%.1fHz' % (foi_phase[0], foi_phase[-1])
else:
    freqs_phase_label = '%.1f-%.1fHz' % (freqs_phase[0], freqs_phase[-1])
if foi_amp and foi_amp[0]==foi_amp[-1]:
    freqs_amp_label = '%dHz' % foi_amp[0]
elif foi_amp and foi_amp[0]!=foi_amp[-1]:
    freqs_amp_label = '%d-%dHz' % (foi_amp[0], foi_amp[-1])
else:
    freqs_amp_label = '%d-%dHz' % (freqs_amp[0], freqs_amp[-1])
out_id = 'phase%s_amp%s' % (freqs_phase_label, freqs_amp_label)
    
for i_cond,cond in enumerate(conds):
    if cond not in corr_conds:
        continue
    for score_label in score_labels:
        brains = data[:, i_cond]
        scores = np.array(sub_info[score_label])
        diagnosed = sub_info['ASD'].copy()
        this_sub_IDs = sub_info['sub_ID'].copy()
        
        # indices of None, i.e. missing scores
        null_inds = [i for i in range(len(scores)) 
                     if scores[i]==None]
        null_inds.reverse() # reverse so that largest index is removed first
        # remove Nones and the corresponding brain data
        for i in null_inds:
            scores = np.delete(scores, i)
            brains = np.delete(brains, i, axis=0)
            diagnosed.pop(i)   
            this_sub_IDs.pop(i)
            
        this_inds_TD = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
        this_inds_ASD = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
            
        if partial and covar_labels:
            #% Regress covariates from scores
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
            
            # zscore scores
            scores = scipy.stats.zscore(scores, axis=None)
        if covar_labels:
            brains = scipy.stats.zscore(brains, axis=None)
            
               
        fn_out = '%s/figures/pac/corr_%s_%s_%s' % (paths['cluster'], cond, 
                                                   mask.name, score_label)
        temp = []
        if len(scores[this_inds_TD])>0:
            for i in inds_mask_fsave:
                temp.append(pearsonr(scores[this_inds_TD], brains[this_inds_TD, i]))
            r_TD = np.array(temp)[:,0]
            p_TD = np.array(temp)[:,1]
            r_TD_th = np.zeros(r_TD.shape)
            r_TD_th[np.where(p_TD < th_p)] = r_TD[np.where(p_TD < th_p)] 
            if plot_correlation:
                for this_data,label in zip([r_TD, r_TD_th], ['TD', 'TD th %s' % th_p]):
                    stc_data = np.zeros(n_verts_fsave)
                    stc_data[inds_mask_fsave] = this_data
                    stc = mne.SourceEstimate(stc_data, vertices=verts_fsave, 
                                              tmin=0, tstep=0, subject='fsaverageJA')
                    max_val = abs(this_data).max()
                    if not max_val: max_val = 1
                    lims = [5e-3, 5e-3, 0.3]
                    brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'], 
                                      hemi='lh', backend='matplotlib', spacing='ico7',
                                      background='k', colorbar=True,
                                      clim=dict(kind='value', pos_lims=lims))
                    brain.text(0.5, 0.8, '%s: %s vs %s' % (label, cond, score_label),  
                                horizontalalignment='center', fontsize=14, color='w')
                    brain.savefig('%s/figures/pac/correlations/%d%s_%s_%s_%s_%s.png' % (paths['cluster'], 
                                                                                      len(this_inds_TD), 
                                                                                      ('_').join(label.split(' ')),
                                                                                      out_id, cond, score_label, 
                                                                                      mask.name), dpi=300)
                    plt.close()                   

        temp = []
        if len(scores[this_inds_ASD])>0:
            for i in inds_mask_fsave:
                temp.append(pearsonr(scores[this_inds_ASD], brains[this_inds_ASD, i]))
            r_ASD = np.array(temp)[:,0]
            p_ASD = np.array(temp)[:,1]
            r_ASD_th = np.zeros(r_ASD.shape)
            r_ASD_th[np.where(p_ASD < th_p)] = r_ASD[np.where(p_ASD < th_p)]
            if plot_correlation:
                for this_data,label in zip([r_ASD, r_ASD_th], ['ASD', 'ASD th %s' % th_p]):
                    stc_data = np.zeros(n_verts_fsave)
                    stc_data[inds_mask_fsave] = this_data
                    stc = mne.SourceEstimate(stc_data, vertices=verts_fsave, 
                                              tmin=0, tstep=0, subject='fsaverageJA')
                    max_val = abs(this_data).max()
                    if not max_val: max_val = 1
                    lims = [5e-3, 5e-3, 0.3]
                    brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'], 
                                      hemi='lh', backend='matplotlib', spacing='ico7',
                                      background='k', colorbar=True,
                                      clim=dict(kind='value', pos_lims=lims))
                    brain.text(0.5, 0.8, '%s: %s vs %s' % (label, cond, score_label),  
                                horizontalalignment='center', fontsize=14, color='w')
                    brain.savefig('%s/figures/pac/correlations/%d%s_%s_%s_%s_%s.png' % (paths['cluster'], 
                                                                                      len(this_inds_ASD), 
                                                                                      ('_').join(label.split(' ')),
                                                                                      out_id, cond, score_label, 
                                                                                      mask.name), dpi=300)
                    plt.close()                   
            
        temp = []
        if len(scores[this_inds_TD])>0 and len(scores[this_inds_ASD])>0:
            for i in range(len(inds_mask_fsave)):
                temp.append(compare_corr_coefs(r_TD[i], r_ASD[i], len(this_inds_TD), 
                                               len(this_inds_ASD)))
            z = np.array(temp)[:,0]
            p = np.array(temp)[:,1]
            z_th = np.zeros(z.shape)
            z_th[np.where(p < th_p)] = z[np.where(p < th_p)]
            if plot_correlation:
                for this_data,label in zip([z, z_th], ['TD vs ASD', 'TD vs ASD th %s' % th_p]):
                    stc_data = np.zeros(n_verts_fsave)
                    stc_data[inds_mask_fsave] = this_data
                    stc = mne.SourceEstimate(stc_data, vertices=verts_fsave, 
                                             tmin=0, tstep=0, subject='fsaverageJA')
                    max_val = abs(this_data).max()
                    if not max_val: max_val = 1
                    lims = [0.1, 0.1, 2]
                    brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'], 
                                     hemi='lh', backend='matplotlib', spacing='ico7',
                                     background='k', colorbar=True,
                                     clim=dict(kind='value', pos_lims=lims))
                    brain.text(0.5, 0.8, '%s: %s vs %s' % (label, cond, score_label),  
                               horizontalalignment='center', fontsize=14, color='w')
                    brain.savefig('%s/figures/pac/correlations/%s_%s_%s_%s_%s.png' % (paths['cluster'], 
                                                                                      ('_').join(label.split(' ')),
                                                                                      out_id, cond, score_label, 
                                                                                      mask.name), dpi=300)
                    plt.close()                   
            
        # if plot_correlation: 
        #     for this_data, group in zip([r_TD, r_ASD, z, r_TD_th, r_ASD_th, z_th], 
        #                                 ['TD', 'ASD', 'TD vs ASD', 'TD th %s' % th_p, 
        #                                  'ASD th %s' % th_p, 'TD vs ASD th %s' % th_p]):
        #         stc_data = np.zeros(n_verts_fsave)
        #         stc_data[inds_mask_fsave] = this_data
        #         stc = mne.SourceEstimate(stc_data, vertices=verts_fsave, 
        #                                  tmin=0, tstep=0, subject='fsaverageJA')
        #         max_val = abs(this_data).max()
        #         if not max_val: max_val = 1
        #         if 'TD vs ASD' in group:                    
        #             lims = [1e-1, 1e-1, max_val*0.8]
        #         else:
        #             lims = [5e-3, 5e-3, max_val*0.8]
        #         brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'], 
        #                          hemi='lh', backend='matplotlib', spacing='ico7',
        #                          background='k', colorbar=True,
        #                          clim=dict(kind='value', pos_lims=lims))
        #         brain.text(0.5, 0.8, '%s: %s vs %s' % (group, cond, score_label),  
        #                    horizontalalignment='center', fontsize=14, color='w')
        #         brain.savefig('%s/figures/pac/correlations/%s_%s_%s_%s_%s.png' % (paths['cluster'], 
        #                                                                           out_id, 
        #                                                                           ('_').join(group.split(' ')), 
        #                                                                           cond, score_label, 
        #                                                                           mask.name), dpi=500)
        #         plt.close()