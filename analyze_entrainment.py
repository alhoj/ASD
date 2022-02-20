#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:08:48 2020

@author: ja151
"""

import mne
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
import math

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
bad_subs = ['105801', '107301']
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

fn_in = 'pdf_0-1600ms_B'
conds = ['SpeechB', 'JabberB', 'MSS', 'SWS', 'NoiseB'] #  
n_conds = len(conds)
rois = ['AC_dSPM_25verts_mean0-1600ms',  'AC_dSPM_50verts_mean0-1600ms',
        'AC_dSPM_100verts_mean0-1600ms',  'AC_dSPM_200verts_mean0-1600ms',
        'bankssts', 'parsorbitalis', 'parstriangularis',
        'transversetemporal', 'lateralorbitofrontal',
        'supramarginal_div1', 'supramarginal_div2', 'supramarginal_div3',
        'inferiorparietal_div1', 'inferiorparietal_div2', 'inferiorparietal_div3',
        'rostralmiddlefrontal_div1', 'rostralmiddlefrontal_div2',
        'rostralmiddlefrontal_div3']
# rois = ['AC_dSPM_25verts_mean0-1600ms',  'AC_dSPM_50verts_mean0-1600ms',
#         'AC_dSPM_100verts_mean0-1600ms',  'AC_dSPM_200verts_mean0-1600ms']
n_rois = len(rois)
hemis = ['lh','rh']
n_hemis = len(hemis)

fmin = 1
fmax = 40
fstep = 1
freqs = np.arange(fmin, fmax+fstep, fstep)
n_freqs = len(freqs)

# regress out covariates
regress_covars = True
covar_labels = ['age', 'NVIQ', 'VIQ'] #

# Read fsaverage source space 
src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
src = mne.read_source_spaces(src_fname)
verts_fsave = [s['vertno'] for s in src]
n_verts_fsave = len(np.concatenate(verts_fsave))
    
    
#%% Load data
data = np.zeros((n_subs, n_conds, n_hemis, n_rois, n_freqs))
for i_sub,sub_ID in enumerate(sub_IDs):  
    print(sub_ID)      
    sub_path = '%s/%s/' % (paths['cluster'], sub_ID)
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    
    for i_hemi,hemi in enumerate(hemis):
        for i_roi,roi in enumerate(rois):
            # try:
            label = mne.read_label('%s/rois/%s-%s.label' % (sub_path, roi, hemi),
                                   subject=fs_id)
            # except:
            #     label = mne.read_label('%s/rois/%s-%s.label' % (paths['cluster'], roi, hemi))
            #     label = label.copy().morph(subject_to=fs_id, subject_from='fsaverageJA', 
            #                                subjects_dir=paths['fs'])
            for i_cond,cond in enumerate(conds):
                stc = mne.read_source_estimate('%s/%s/%s_%s' 
                                               % (paths['cluster'], 
                                                  sub_ID, fn_in, cond),
                                               subject=fs_id)        
                data[i_sub, i_cond, i_hemi, i_roi] = stc.in_label(label).data.squeeze().mean(0)
        
data_orig = data

#%% Group average peak freq data

group_specific_peaks = True

data = np.zeros((n_subs, n_conds, n_hemis, n_rois))
for i_hemi,hemi in enumerate(hemis):
    for i_roi,roi in enumerate(rois):
        for i_cond,cond in enumerate(conds):
            peak_freq_all = freqs[np.argmax(data_orig.mean(0)[i_cond, i_hemi, i_roi], axis=-1)]
            peak_freq_TD = freqs[np.argmax(data_orig[inds_TD].mean(0)[i_cond, i_hemi, i_roi], axis=-1)]
            peak_freq_ASD = freqs[np.argmax(data_orig[inds_ASD].mean(0)[i_cond, i_hemi, i_roi], axis=-1)]
            ind_peak_freq_all = list(freqs).index(peak_freq_all)
            ind_peak_freq_TD = list(freqs).index(peak_freq_TD)
            ind_peak_freq_ASD = list(freqs).index(peak_freq_ASD)
            if group_specific_peaks:
                data[inds_TD,i_cond,i_hemi,i_roi] = data_orig[inds_TD,i_cond,i_hemi,i_roi,ind_peak_freq_TD]
                data[inds_ASD,i_cond,i_hemi,i_roi] = data_orig[inds_ASD,i_cond,i_hemi,i_roi,ind_peak_freq_ASD]
            else:
                data[:,i_cond,i_hemi,i_roi] = data_orig[:,i_cond,i_hemi,i_roi,ind_peak_freq_all]

#%% Plot entrainment
    
group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
group_labels = np.repeat(group_labels, n_conds)
cond_labels = conds * n_subs

c_TD = 'lightgreen'
c_ASD = 'orchid'

fn_id = fn_in.replace('B','GAgroupSpecificPeakFreq')    

for i_hemi,hemi in enumerate(hemis):
    for i_roi,roi in enumerate(rois):
        data_pd = {'Entrainment': np.concatenate(data[:,:,i_hemi,i_roi]), 
                   'Group': group_labels,
                   'Condition': cond_labels,
                   'ID': np.repeat(sub_IDs, n_conds)}
        
        data_frame = pd.DataFrame(data=data_pd)
        
        plt.figure()
        sns.barplot(x='Condition', y='Entrainment', hue='Group', data=data_frame,
                      alpha=1, ci=68, edgecolor='gray', palette=[c_ASD,c_TD])
        # sns.stripplot(x='Condition', y='Entrainment', hue='Group', data=data_frame,
        #               alpha=1, edgecolor='gray', palette=[c_ASD,c_TD])
        # sns.violinplot(x='Condition', y='Entrainment', hue='Group', data=data_frame,
        #               cut=0, inner='point', alpha=1, ci=68, edgecolor='gray', 
        #               palette=[c_ASD,c_TD])
        plt.title('%s - %s' % (roi, hemi))
        plt.rcParams['axes.labelsize'] = 14
        plt.savefig('%s/figures/entrainment/bars_%s_%s-%s.png' % (paths['cluster'], 
                                                                  fn_id, roi, hemi), dpi=300)
        plt.close()
#%% Regress covariates out from the data

if regress_covars:
    print('Regressing out covariates: %s' % covar_labels)
    covars = []
    for covar_label in covar_labels:
        covars.append(sub_info[covar_label])
    model = np.zeros((n_subs,len(covars)+1))
    model[:,0] = np.ones(n_subs)
    model[:,1::] = np.transpose(covars)
       
    data = np.reshape(data, (n_subs, n_conds*n_hemis*n_rois))
    beta = scipy.linalg.lstsq(model, data)[0]
    data = data - model.dot(beta)
    data = np.reshape(data, (n_subs, n_conds, n_hemis, n_rois))

else: 
    print('No covariates')
    covar_labels = ['no_covars']
    
#%% Entrainment t-tests 

# t-tests
for i_hemi,hemi in enumerate(['lh', 'rh']):
    for i_roi,roi in enumerate(rois):
        for i_cond,cond in enumerate(conds):
            stats = pg.ttest(data[inds_TD, i_cond, i_hemi, i_roi], 
                             data[inds_ASD, i_cond, i_hemi, i_roi], paired=False)
            print('T-test TD vs. ASD - %s - %s - %s: t=%.2f, p=%.2f' % (cond, roi, 
                                                                        hemi,
                                                                        stats['T'][0],
                                                                        stats['p-val'][0]))
#%% Entrainment ANOVA 

c_TD = 'lightgreen'
c_ASD = 'orchid'
    
conds_anova = ['SWS', 'MSS']
inds_cond = [conds.index(cond_anova) for cond_anova in conds_anova]    

group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
group_labels = np.repeat(group_labels, len(conds_anova))
cond_labels = conds_anova * n_subs

for i_hemi,hemi in enumerate(['lh', 'rh']):
    data_pd = {'Entrainment': np.concatenate(data[:,inds_cond,i_hemi]), 
               'Group': group_labels,
               'Condition': cond_labels,
               'ID': np.repeat(sub_info['sub_ID'],len(conds_anova))}
    
    data_frame = pd.DataFrame(data=data_pd)
    
    # plot
    plt.figure()
    sns.barplot(x='Condition', y='Entrainment', hue='Group', data=data_frame,
                alpha=1, ci=68, edgecolor='gray', palette=[c_ASD,c_TD])
    
    #% ANOVA
    stats = pg.mixed_anova(data=data_frame, dv='Entrainment', between='Group', 
                             within='Condition', subject='ID')
    print('ANOVA group x condition interaction - %s: \
          F=%f, p=%f, np2=%f' % (hemi, stats['F'][2], stats['p-unc'][2], 
                                  stats['np2'][2]))
    
#%% Entrainment ANOVA (interhemispheric)
    
c_TD = 'lightgreen'
c_ASD = 'orchid'
    
cond_anova = 'SWS'
ind_cond = conds.index(cond_anova)
    
group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
group_labels = np.repeat(group_labels, 2)
hemi_labels = ['lh', 'rh'] * n_subs

data_pd = {'Entrainment': np.concatenate(data[:,ind_cond,:]), 
           'Group': group_labels,
           'Hemisphere': hemi_labels,
           'ID': np.repeat(sub_info['sub_ID'], 2)}

data_frame = pd.DataFrame(data=data_pd)

# plot
plt.figure()
sns.barplot(x='Hemisphere', y='Entrainment', hue='Group', data=data_frame,
            alpha=1, ci=68, edgecolor='gray', palette=[c_ASD,c_TD])

#% ANOVA
stats = pg.mixed_anova(data=data_frame, dv='Entrainment', between='Group', 
                         within='Hemisphere', subject='ID')
print('ANOVA group x hemisphere interaction: \
      F=%f, p=%f, np2=%f' % (stats['F'][2], stats['p-unc'][2], 
                              stats['np2'][2]))
    
#%% Entrainment correlations
    
plot = False
    
print('Correlation between entrainment and behavioral scores\n')
score_labels = ['age', 'NVIQ', 'VIQ', 'SRS_tot_T', 'ASPS', 'ICSS', 'SCSS'] #  
for i_hemi,hemi in enumerate(hemis):
    for i_roi,roi in enumerate(rois):
        
        for i_cond,cond in enumerate(conds):
            for score_label in score_labels:
                for inds_group,group in zip([inds_TD, inds_ASD], ['TD', 'ASD']):
                    
                    if group=='TD': 
                        color = 'lightgreen'
                    else:
                        color = 'orchid'        
                
                    brains = list(data[inds_group, i_cond, i_hemi, i_roi])
                    scores = sub_info[score_label]
                    scores = list(np.array(scores)[inds_group])
                    # indices of None, i.e. missing scores
                    inds_null = [i for i in range(len(scores)) if scores[i]==None]
                    inds_null.reverse() # reverse so that largest are removed first
                    # remove Nones and the corresponding brain data value
                    for i in inds_null:
                        scores.pop(i)
                        brains.pop(i)
                        
                    if scores:
                        r, p = scipy.stats.pearsonr(brains, scores)
                        if p<0.05:
                            print('\n%s - %s' % (roi, hemi))
                            print('Group - %s' % group)
                            print('%s - %s: r=%.2f, p=%.3f' % (cond, score_label,
                                                                r, p))
                    # else:
                    #     print('No %s scores for %s' % (score_label, group))
                        
                    if plot:
                        sns.regplot(scores, brains, ci=None, scatter_kws={'s':80}, 
                                    color=color, marker='o')
                if plot:
                    plt.xlabel(score_label)
                    plt.ylabel('Entrainment')
                    plt.savefig('%s/figures/entrainment/correlation_%s_%s_%s_%s-%s.png' % (paths['cluster'], 
                                                                                   cond, score_label,  
                                                                                   fn_in, roi, hemi), dpi=300)
                    plt.close()
                    
#%% Laterality index
    
# values have to positive
min_val = np.min(data)
if min_val < 0:
    data = data + abs(min_val)

LI = []
for i_roi in range(n_rois):
    LI.append((data[:, :, 1, i_roi] - data[:, :, 0, i_roi]) / 
              (data[:, :, 1, i_roi] + data[:, :, 0, i_roi]))
    
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
    plt.savefig('%s/figures/entrainment/LI_%s_%s.png' % (paths['cluster'], 
                                                         fn_in, roi), dpi=300)
    plt.close()
        
#%% LI t-tests 

# t-tests
for i_cond,cond in enumerate(conds):
    stats = pg.ttest(LI[inds_TD, i_cond], LI[inds_ASD, i_cond], paired=False)
    print('T-test TD vs. ASD - %s: t=%.2f, p=%.2f' % (cond, stats['T'][0],
                                                      stats['p-val'][0]))
#%% LI ANOVA
    
conds_anova = ['SWS', 'MSS']
cond_inds = [conds.index(cond_anova) for cond_anova in conds_anova]
    
group_labels = [None] * n_subs
for i in inds_TD: group_labels[i] = 'TD'
for i in inds_ASD: group_labels[i] = 'ASD'
group_labels = np.repeat(group_labels, len(conds_anova))
cond_labels = conds_anova * n_subs

data_pd = {'LI': np.concatenate(LI[:,cond_inds]), 
           'Group': group_labels,
           'Condition': cond_labels,
           'ID': np.repeat(sub_info['sub_ID'], len(conds_anova))}

data_frame = pd.DataFrame(data=data_pd)

#% ANOVA
stats = pg.mixed_anova(data=data_frame, dv='LI', between='Group', 
                         within='Condition', subject='ID')
print('ANOVA group x condition interaction: F=%f, p=%f, np2=%f' % (stats['F'][2], 
                                                                    stats['p-unc'][2],
                                                                    stats['np2'][2]))


#%% LI correlations

plot = False    

print('Correlation between laterality index and behavioral scores\n')
score_labels = ['SRS_tot_T', 'ASPS', 'ICSS', 'SCSS'] # 'age', 'VIQ', 'NVIQ', 
for i_roi,roi in enumerate(rois):
    for i_cond,cond in enumerate(conds): 
        for score_label in score_labels: 
            scores = sub_info[score_label]
            xmin = np.min(list(filter(None, scores)))
            xmax = np.max(list(filter(None, scores)))
            ymin = np.min(LI[i_roi, :, i_cond])
            ymax = np.max(LI[i_roi, :, i_cond])
            for inds_group,group in zip([inds_TD, inds_ASD], ['TD', 'ASD']):
                
        
                brains = list(LI[i_roi, inds_group, i_cond])
                scores = sub_info[score_label]
                scores = list(np.array(scores)[inds_group])
                # indices of None, i.e. missing scores
                inds_null = [i for i in range(len(scores)) if scores[i]==None]
                inds_null.reverse() # reverse so that largest are removed first
                # remove Nones and the corresponding brain data value
                for i in inds_null:
                    scores.pop(i)
                    brains.pop(i)
                    
                if scores:
                    r, p = scipy.stats.pearsonr(brains, scores)
                    if p<0.1:
                        print('\nGroup - %s' % group)
                        print('%s - %s - %s: r=%.2f, p=%.4f' % (roi, cond, 
                                                                score_label, r, p))
                # else:
                #     print('No %s scores for %s' % (score_label, group))
                    
                if group=='TD':
                    color = 'lightgreen'
                    n_TD = len(scores)
                    r_TD = r
                    p_TD = p
                    z_TD = math.atanh(r)
                else:
                    color = 'orchid'
                    n_ASD = len(scores)
                    r_ASD = r
                    p_ASD = p
                    z_ASD = math.atanh(r)
                
                if scores and plot:
                    sns.regplot(scores, brains, ci=None, scatter_kws={'s':80}, 
                                color=color, marker='o')
                        
            # expected standard deviation
            sd = np.sqrt(1/(n_TD-3) + 1/(n_ASD-3))            
            # z-score and p-value for the difference between the correlations
            z = abs((z_TD - z_ASD)/sd)
            p = (1 - scipy.stats.norm.cdf(z)) * 2
            if p<0.05:
                print('\n%s - %s - %s - difference between groups: z=%f, p=%f' 
                      % (roi, cond, score_label, z,p))
            
            if scores and plot:
                ax = plt.gca()
                ax.set_xlim(xmin-(xmax-xmin)*0.1,xmax+(xmax-xmin)*0.1)
                ax.set_ylim(ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1)
                plt.xlabel(score_label)
                plt.ylabel('LI')
                ax.text(0.05, 0.95,'r=%.2f, p=%.2f' % (r_TD, p_TD), 
                         color='lightgreen', transform=ax.transAxes)
                ax.text(0.05, 0.9,'r=%.2f, p=%.2f' % (r_ASD, p_ASD), 
                         color='orchid', transform=ax.transAxes)
                ax.text(0.05, 0.85,'z=%.2f, p=%.2f' % (z, p), 
                         color='k', transform=ax.transAxes)
                plt.savefig('%s/figures/entrainment/LI_correlation_%s_%s_%s_%s-%s.png' % (paths['cluster'], 
                                                                               cond, score_label,  
                                                                               fn_in, roi, hemi), dpi=300)
                plt.close()