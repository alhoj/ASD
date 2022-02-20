#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:00:25 2020

@author: ja151
"""


import mne
import pickle
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, ranksums, pearsonr
import pandas as pd
import seaborn as sns
from helper_functions import compare_corr_coefs
import itertools

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

f = open('%s/p/subjects.p' % paths['out'], 'rb')
sub_info = pickle.load(f)
f.close()

# define bad subjects
# bad_subs = ['105801', '107301']
# equalized NVIQs
# bad_subs = ['105801', '107301', '052902', '090902', '048102']
# equalized NVIQs - new sample N_ATD=28, N_ASD=28
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

fn_in = 'con_-500-1500ms_8-12Hz_MSS-SWS_seedROI_AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh_wpli2_debiased.p'

covar_labels = [] # 'age', 'NVIQ', 'VIQ'
tmin = 0. 
tmax = 1.5 # times of interest 
fmin = 8
fmax = 12 # frequencies of interest 
average_times = True # if not averaged already
decimate = False # by factor
add_bl_cond = True # if baseline period included

# load one to get parameters
f = open('%s/%s/p/%s' % (paths['cluster'], sub_IDs[0], fn_in), 'rb')
data = pickle.load(f)
f.close()

conds = data['conds']
n_conds = len(conds)
rois = data['rois']
n_rois = len(rois)
rois_as_seed = data['rois_as_seed']

if rois_as_seed:      
    n_cons = 8196
else:
    con_labels = data['con_labels']
    n_cons = len(con_labels)
    # con_labels = [('-').join(pair) for pair in itertools.combinations(rois, 2)]
    
times = data['times']
if type(times)==tuple:
    times = np.round(np.arange(times[0], times[1]+1e-3, 1e-3), 3)
if decimate:
    times = times[0:-1:decimate]
if tmax > times[-1]: tmax = times[-1]
i_tmin = list(times).index(tmin)
i_tmax = list(times).index(tmax)
times = times[i_tmin:i_tmax+1]
n_times = len(times)
times_averaged = data['average_times']

freqs = data['freqs']
i_fmin = list(freqs).index(fmin)
i_fmax = list(freqs).index(fmax)
freqs = freqs[i_fmin:i_fmax+1]
n_freqs = len(freqs) 
freqs_averaged = data['average_freqs']

con_methods = data['con_method']
n_con_methods = len(con_methods)

if rois_as_seed:
    # Read fsaverage source space 
    src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
    src_fsave = mne.read_source_spaces(src_fname)
    verts_fsave = [s['vertno'] for s in src_fsave]
    n_verts_fsave = len(np.concatenate(verts_fsave))

#% Load data
print('Loading data...')
if average_times:
    data = np.zeros((n_subs, n_conds, n_con_methods, n_cons))
    if add_bl_cond:
        data_bl = np.zeros(np.shape(data))
else:
    data = np.zeros((n_subs, n_conds, n_con_methods, n_cons, n_times))
    
    
 
#%%
for i_sub,sub_ID in enumerate(sub_IDs):
    print(sub_ID)
    sub_path = '%s/%s' % (paths['cluster'], sub_ID)
    
    if rois_as_seed:
        fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
        inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                                     % (sub_path, sub_ID))
        src = inv['src']
        verts = [s['vertno'] for s in src]
   
    f = open('%s/p/%s' % (sub_path, fn_in), 'rb')
    data_sub = pickle.load(f)
    f.close()
            
    # put connectivity data in array
    for i_cond, cond in enumerate(conds):
        for i_con_method, con_method in enumerate(con_methods):
            if n_con_methods>1:
                temp = data_sub['cons'][cond][i_con_method]
            else:
                temp = data_sub['cons'][cond]
            if not times_averaged:
                if decimate:
                    temp = temp[...,0:-1:decimate]
                temp = temp[...,i_tmin:i_tmax+1]
                if average_times:
                    temp = temp.mean(-1)                    
            if not freqs_averaged:
                temp = temp[:,i_fmin:i_fmax+1].mean(1)

            if rois_as_seed:
                stc = mne.SourceEstimate(temp, vertices=verts, tmin=0, 
                                         tstep=1e-3, subject=fs_id)
                # morph individual source estimate to freesurfer average brain
                morph = mne.compute_source_morph(src, fs_id, 'fsaverageJA',
                                                 spacing=verts_fsave, 
                                                 subjects_dir=paths['fs'])
                stc = morph.apply(stc)
                data[i_sub, i_cond, i_con_method] = stc.data.squeeze()
            else:
                # take all connections between ROIs
                # cons_mask = np.array(np.tril(np.ones((n_rois, n_rois)), k=-1), dtype=bool)
                data[i_sub, i_cond, i_con_method] = temp
                
    if add_bl_cond:
        for i_cond, cond in enumerate(conds):
            for i_con_method, con_method in enumerate(con_methods):
                if n_con_methods>1:
                    temp = data_sub['cons'][cond][i_con_method]
                else:
                    temp = data_sub['cons'][cond]
                if not times_averaged:
                    if decimate:
                        temp = temp[...,0:-1:decimate]
                    temp = temp[...,0:i_tmin]
                    if average_times:
                        temp = temp.mean(-1)                    
                if not freqs_averaged:
                    temp = temp[:,i_fmin:i_fmax+1].mean(1)
    
                if rois_as_seed:
                    stc = mne.SourceEstimate(temp, vertices=verts, tmin=0, 
                                             tstep=1e-3, subject=fs_id)
                    # morph individual source estimate to freesurfer average brain
                    morph = mne.compute_source_morph(src, fs_id, 'fsaverageJA',
                                                     spacing=verts_fsave, 
                                                     subjects_dir=paths['fs'])
                    stc = morph.apply(stc)
                    data_bl[i_sub, i_cond, i_con_method] = stc.data.squeeze()
                else:
                    # take all connections between ROIs
                    # cons_mask = np.array(np.tril(np.ones((n_rois, n_rois)), k=-1), dtype=bool)
                    data_bl[i_sub, i_cond, i_con_method] = temp


#%% Contrasts
                    
data_contrasts = []
contrasts = []
for i_cond,cond1 in enumerate(conds[0:-1]):
    for j_cond,cond2 in enumerate(conds[i_cond+1::], i_cond+1):
        data_contrasts.append(data[:, i_cond] - data[:, j_cond])
        contrasts.append('%s-%s' % (cond1, cond2))

if add_bl_cond:
    for i_cond,cond in enumerate(conds):
        data_contrasts.append(data[:, i_cond] - data_bl.mean(1))
        contrasts.append('%s-BL' % cond)
        
dims = np.arange(len(np.shape(data_contrasts)))
dims_transposed = [dims[1]]+[dims[0]]+list(dims[2::]) # swap first two
data_contrasts = np.array(data_contrasts).transpose(dims_transposed)

# add contrasts data
data = np.concatenate((data, data_contrasts), axis=1)
del data_contrasts
conds = conds + contrasts
n_conds = len(conds)

#% Regress covariates    
if covar_labels:
    print('Regressing out covariates: %s' % covar_labels)
    covar_out = '%sReg' % ('').join(covar_labels)
    covars = []
    for covar_label in covar_labels:
        covars.append(sub_info[covar_label])
    model = np.zeros((n_subs,len(covars)+1))
    model[:,0] = np.ones(n_subs)
    model[:,1::] = np.transpose(covars)
       
    # data = np.transpose(data_orig, (1,0,2,3,4))
    data = np.reshape(data, (n_subs, n_conds*n_con_methods*n_cons))
    beta = scipy.linalg.lstsq(model, data)[0]
    data = data - model.dot(beta)
    # intercept = beta[0]
    # data += intercept
    data = np.reshape(data, (n_subs, n_conds, n_con_methods, n_cons))
    # data = np.transpose(data, (1,0,2,3,4))

else: 
    # data = data_orig
    print('No covariates')
    covar_out = 'noReg'

#%% group comparison on ROI-to-ROI connections

con_method = 'wpli2_debiased'
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']

print('T-test TD vs. ASD:')
for i_con, con_label in enumerate(con_labels):
    for i_cond, cond in enumerate(conds):    
        results = pg.ttest(data[inds_TD, i_cond, con_methods.index(con_method), i_con], 
                           data[inds_ASD, i_cond, con_methods.index(con_method), i_con], paired=False)
        p = results['p-val'][0]
        t = results['T'][0]
        if p<0.05:
            print('\nConnection: %s' % con_label)
            print('%s: t=%.2f, p=%.4f\n' % (cond, t, p))


#%% bar plot
                 
con_label = 'AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh - IP_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh'
cois = ['MSS', 'SWS', 'MSS-SWS']
con_method = 'wpli2_debiased'
nonparametric = True # two-sample t-test of nonparametric wilcoxon rank sums
# cluster_mask = np.load('%s/ACseedConn_clusterMask.npy' % paths['cluster'])

group_labels = sub_info['ASD'] * len(cois)
# replace 'Yes' with 'ASD' and 'No' with 'TD' in group labels
group_labels = ['ASD' if group_labels[i]=='Yes' else 'TD' 
                for i in range(len(group_labels))]
cond_labels = np.repeat(cois, n_subs).tolist()

inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
# inds_cois = [i for i,cond in enumerate(conds) if cond in cois]
# this_data = data[:, inds_cois, con_labels.index(con_label)]

this_data = {coi: data[:, conds.index(coi), con_methods.index(con_method), 
                       con_labels.index(con_label)]
              for coi in cois}
data_pd = {'Connectivity (%s)' % con_method: np.concatenate(list(this_data.values())), 
           'Group': group_labels,
           'Condition': cond_labels,
           'ID': sub_info['sub_ID'] * len(cois)}
data_frame = pd.DataFrame(data=data_pd)

c_TD = 'lightgreen'
c_ASD = 'orchid'
plt.figure()
ax = sns.barplot(x='Condition', y='Connectivity (%s)' % con_method, hue='Group', 
                 data=data_frame, alpha=1, edgecolor='gray', ci=68, 
                 palette=[c_TD, c_ASD], hue_order=['TD', 'ASD'])
con_label_out = '%s-%s' % (con_label[0:2], con_label.split(' - ')[1][0:2])
plt.title('%s %d-%dHz' % (con_label_out, fmin, fmax))
plt.legend(loc='lower right')

stat, p = [], []
for i_coi,coi in enumerate(cois):
    if nonparametric:
        results = ranksums(this_data[coi][inds_TD], this_data[coi][inds_ASD])
    else:
        results = ttest_ind(this_data[coi][inds_TD], this_data[coi][inds_ASD])
    stat.append(results[0])
    p.append(results[1])
                
signifs = []
for val in p:
    if val>0.06: signifs.append('')
    elif val<0.06 and val>0.05: signifs.append('(*)')
    elif val<0.05 and val>0.01: signifs.append('*')
    elif val<0.01 and val>0.001: signifs.append('**')
    elif val<0.001: signifs.append('***')
    
yloc = ax.get_ybound()[1] - (ax.get_ybound()[1] * 0.2)
for signif, xloc in zip(signifs, ax.get_xticks()):
    ax.text(xloc, yloc, signif, fontsize=20,
            horizontalalignment='center') 
plt.tight_layout()
fn_out = '%s/figures/connectivity/%dTD_%dASD_bar_%s_%s_%d-%dms_%d-%dHz_%s' \
        % (paths['cluster'], n_TD, n_ASD, con_method, con_label_out, 
           int(tmin*1e3), int(tmax*1e3), 
               fmin, fmax, ('_').join(cois))
if nonparametric:
    fn_out += '_ranksum'
else:
    fn_out += '_ttest'
fn_out += '_%sReg' % covar_out
plt.savefig('%s.png' % fn_out, dpi=300)
plt.close()

#%% behavioral correlations

cois = ['MSS-BL', 'SWS-BL', 'MSS-SWS'] # 'MSS', 'SWS',  
con_method = 'wpli'
con_label = 'AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh - IP_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh'
score_labels = ['age']
    # 'ASPS', 'ASPSa8', 'ASPSa1-a5', 'ASPSa1-a7', 'ASPSa6-a7', 
    #             'ICSS', 'SCSS', 'ADOS_tot_old', 'ADOS_comm_soc_old', 'SRS_tot_T']

plot = False
save_brains = True

c_TD = 'lightgreen'
c_ASD = 'orchid'

con_label_out = '%s-%s' % (con_label[0:2], con_label.split(' - ')[1][0:2])
for coi in cois:
    if save_brains:
        brains = data[:, conds.index(coi), con_methods.index(con_method), 
                      con_labels.index(con_label)]
        # inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
        # inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
        np.save('%s/npy/%s_N%d_%d-%dHz_%s_%s_%s' % (paths['cluster'], con_method, 
                                                     n_subs, fmin, fmax, 
                                                     coi, con_label_out, 
                                                     covar_out), brains)
        # np.save('%s/npy/%dASD_%s_%d-%dHz_%s_%s_%s' % (paths['cluster'], n_ASD, 
        #                                               con_method, fmin, fmax, 
        #                                               coi, con_label_out, 
        #                                               covar_out), brains[inds_ASD])
    for score_label in score_labels:
        brains = data[:, conds.index(coi), con_methods.index(con_method), 
                      con_labels.index(con_label)]
        scores = np.array(sub_info[score_label])
        
        diagnosed = sub_info['ASD'].copy()    
        # indices of None, i.e. missing scores
        null_inds = [i for i in range(len(scores)) if scores[i]==None]
        null_inds.reverse() # reverse so that largest index is removed first
        # remove Nones and the corresponding brain data
        for i in null_inds:
            scores = np.delete(scores, i)
            brains = np.delete(brains, i, axis=0)
            diagnosed.pop(i)   
            
        inds_TD = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
        inds_ASD = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
               
        fn_out = '%s/figures/connectivity/correlations/%dTD_%dASD_%s_%s_%d-%dms_%d-%dHz_%s_%s_%s' \
                    % (paths['cluster'], n_TD, n_ASD, con_method, con_label_out, int(tmin*1e3), 
                       int(tmax*1e3), int(fmin), int(fmax), coi, score_label, covar_out)
    
        if len(scores[inds_TD])>0:
            r_TD, p_TD = pearsonr(scores[inds_TD], brains[inds_TD])
            if p_TD < 0.05:
                print('\nTD: %s vs. %s: r=%f, p=%f' % (coi, score_label, r_TD, p_TD))                    
    
        if len(scores[inds_ASD])>0:
            r_ASD, p_ASD = pearsonr(scores[inds_ASD], brains[inds_ASD])
            if p_ASD < 0.05:
                print('\nASD: %s vs. %s: r=%f, p=%f' % (coi, score_label, r_ASD, p_ASD))
            
        if len(scores[inds_TD])>0 and len(scores[inds_ASD])>0:
            z,p = compare_corr_coefs(r_TD, r_ASD, len(inds_TD), len(inds_ASD))
            if p < 0.05:
                print('\nTD vs. ASD: %s vs. %s: z=%f, p=%f' % (coi, score_label, z, p))
            
        if plot and (p_ASD<0.05 or p_TD<0.05 or p<0.05):
            plt.figure()
            if len(scores[inds_TD])>0:
                ax = sns.regplot(list(scores[inds_TD]), brains[inds_TD], ci=None, 
                            scatter_kws={'s':80}, color=c_TD, marker='o')
                ax.text(0.025, 0.925, 'r=%.2f, p=%.4f' % (r_TD, p_TD), fontsize=12,                                    
                        color=c_TD, transform=ax.transAxes)
            if len(scores[inds_ASD])>0:
                ax = sns.regplot(list(scores[inds_ASD]), brains[inds_ASD], ci=None, 
                            scatter_kws={'s':80}, color=c_ASD, marker='s')
                ax.text(0.025, 0.875, 'r=%.2f, p=%.4f' % (r_ASD, p_ASD), fontsize=12,                                    
                        color=c_ASD, transform=ax.transAxes)     
                
            if len(scores[inds_TD])>0 and len(scores[inds_ASD])>0:
                ax.text(0.025, 0.825, 'z=%.2f, p=%.4f' % (z, p), fontsize=12,                                    
                        color='k', transform=ax.transAxes) 
            
            ax.set_xlim(min(scores)-(max(scores)-min(scores))*0.05, 
                        max(scores)+(max(scores)-min(scores))*0.05)
            ax.set_ylim(min(brains)-(max(brains)-min(brains))*0.05, 
                        max(brains)+(max(brains)-min(brains))*0.05)
            plt.title('%s %s' % (coi, con_label_out))
            plt.xlabel('%s' % score_label)
            plt.ylabel('Connectivity (%s)' % con_method) 
            plt.tight_layout()
            plt.savefig('%s.png' % fn_out, dpi=300)
            plt.close()

#%% permutation cluster test between groups on seed-ROI to the rest of the brain connectivity

cluster_conds = ['MSS-SWS']  
roi = None #'pac_phase8.0-12.0Hz_amp30-60Hz_28TD_vs_28ASD_MSS-SWS_cdt0.01-lh.label'
con_method = 'wpli2_debiased'
hemis = ['lh'] #, 'rh'

n_perms = 5000
cdt_p = 0.01
spatiotemporal = False
n_jobs = 12

plot_cluster = False
bg = 'w'
pht_p = None #1e-6, 1e-7, 1e-8, 1e-9, 1e-10] # post-hoc thresholding the significant cluster
save_roi = False # save largest cluster as ROI

do_corr = True
save_brains = True
corr_conds = ['MSS-BL', 'MSS-SWS']
score_labels = ['age', 'ICSS', 'SCSS', 'ASPS', 'ASPSa1-a7', 'ASPSa1-a5', 
                'SRS_tot_T', 'SRS_socComm_T', 'ADOS_tot_old', 'ADOS_comm_soc_old'] 
mode = 'mean' # 'peak' or 'mean'
plot_corr = False

mean_plot = False #'violin' # 'bar', 'violin'
nonparametric = True # two-sample t-test of nonparametric wilcoxon rank sums
plot_conds = ['SWS-BL']
mark_signifs = True

roi_label = fn_in.split('_')[5]

inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']


# min_pvals = np.zeros((n_con_methods, len(cois), len(hemis)))
for i_hemi,hemi in enumerate(hemis):
    print('\nHemisphere: %s\n' % hemi)
    # Define regions to exclude from the analysis
    if hemi == 'lh':
        labels = mne.read_labels_from_annot('fsaverageJA', parc='PALS_B12_Lobes', hemi='both',
                                           subjects_dir=paths['fs'], 
                                           regexp='MEDIAL.WALL|LOBE.OCCIPITAL|LOBE.LIMBIC|' + 
                                           'LOBE.FRONTAL-rh|LOBE.PARIETAL-rh|LOBE.TEMPORAL-rh')
    elif hemi == 'rh':
        labels = mne.read_labels_from_annot('fsaverageJA', parc='PALS_B12_Lobes', hemi='both',
                                           subjects_dir=paths['fs'], 
                                           regexp='MEDIAL.WALL|LOBE.OCCIPITAL|LOBE.LIMBIC|' +
                                           'LOBE.FRONTAL-lh|LOBE.PARIETAL-lh|LOBE.TEMPORAL-lh')
    elif hemi=='both':
        labels = mne.read_labels_from_annot('fsaverageJA', parc='PALS_B12_Lobes', hemi='both',
                                           subjects_dir=paths['fs'], 
                                           regexp='MEDIAL.WALL|LOBE.OCCIPITAL|LOBE.LIMBIC')
    
    inds_exclude = []
    for label in labels:
        if label.hemi == 'lh':
            verts_label = label.get_vertices_used(verts_fsave[0])
            inds_exclude.append(np.searchsorted(verts_fsave[0], verts_label))
        else:
            verts_label = label.get_vertices_used(verts_fsave[1])
            inds_exclude.append(len(verts_fsave[0])+
                                   np.searchsorted(verts_fsave[1], verts_label))
    inds_exclude = np.sort(np.hstack(inds_exclude).tolist())
    exclude = np.zeros(n_verts_fsave, dtype=bool) 
    exclude[inds_exclude] = 1
              
    if cdt_p=='TFCE':
        th = dict(start=0, step=1)
    else:
        th = scipy.stats.distributions.f.ppf(1. - cdt_p / 2., 
                                                n_ASD - 1, n_TD - 1)        
    adjacency = mne.spatial_src_adjacency(src_fsave)
    
    # for i_con_method, con_method in enumerate(con_methods):
    #% permutation cluster test
    i_coi = 0
    for i_cond,cond in enumerate(conds):
        if cond not in cluster_conds:
            continue
        if not roi:
            print('\nCalculating two-sample permutation cluster test '
                  '%d TD vs. %d ASD: %s' % (n_TD, n_ASD, cond))  
            X = [data[inds_TD, i_cond, con_methods.index(con_method)], 
                 data[inds_ASD, i_cond, con_methods.index(con_method)]]
            if spatiotemporal:
                X = np.transpose(X, (0,1,3,2))
                stat_vals, clusters, cluster_pvals, H0 = \
                        mne.stats.spatio_temporal_cluster_test(X, adjacency=adjacency, 
                                                                spatial_exclude=inds_exclude,
                                                                n_permutations=n_perms+1, 
                                                                n_jobs=n_jobs, threshold=th,
                                                                out_type='mask')
            else:
                stat_vals, clusters, cluster_pvals, H0 = \
                            mne.stats.permutation_cluster_test(X, adjacency=adjacency, 
                                                                exclude=exclude,
                                                                n_permutations=n_perms+1, 
                                                                n_jobs=n_jobs, threshold=th,
                                                                out_type='mask', seed=8)  
                            
            print('Smallest cluster p-value: %.3f' % min(cluster_pvals))
            good_cluster_inds = np.where(cluster_pvals < 0.05)[0]
            cluster_mask = np.zeros(n_verts_fsave, dtype=bool)        
            # min_pvals[i_con_method, i_coi, i_hemi] = min(cluster_pvals)
            i_coi += 1
            if len(good_cluster_inds) > 0:
                stat_vals_th = np.zeros(n_verts_fsave)
                if cdt_p=='TFCE':
                    stat_vals_th[good_cluster_inds] = stat_vals[good_cluster_inds]
                else:
                    for good_cluster_ind in range(len(good_cluster_inds)):
                        stat_vals_th[clusters[good_cluster_ind]] = stat_vals[clusters[good_cluster_ind]]
                # for pht_p in pht_ps:
                if pht_p:
                    th = scipy.stats.distributions.f.ppf(1. - pht_p / 2., 
                                                         n_ASD - 1, n_TD - 1)  
                    stat_vals_th[np.where(stat_vals_th < th)] = 0
    
                cluster_mask[np.nonzero(stat_vals_th)] = True  
                
            stc = mne.SourceEstimate(stat_vals_th, vertices=verts_fsave, tmin=0, 
                                      tstep=0, subject='fsaverageJA')
            if plot_cluster:
                max_val = np.max(stc.data)
                brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'],
                                 hemi=hemi, backend='matplotlib', background=bg, spacing='ico7',
                                 clim=dict(kind='value', lims=[0.0001, 0.0001, 15]))
                plt.title('F statistic', color='w')
                brain.text(0.5, 0.8, 'p=%.3f' % np.min(cluster_pvals), 
                            horizontalalignment='center', fontsize=14, color='w')
                brain.savefig(
                    '%s/figures/connectivity/clusterStats_%s_%d-%dHz_%dTD_vs_%dASD_%s_seed%s_cdt%s_pht%s-%s_%s.png' \
                              % (paths['cluster'], con_method, int(freqs[0]), 
                                 int(freqs[-1]), n_TD, n_ASD, cond, roi_label, 
                                 cdt_p, pht_p, hemi, bg), dpi=500)
                plt.close()
                
            if save_roi:
                labels = mne.stc_to_label(stc, src_fsave, smooth=True, 
                                          connected=True, subjects_dir=paths['fs'])[0]
                n_verts = 0
                for i,this_label in enumerate(labels):
                    temp = len(this_label.get_vertices_used())
                    if temp > n_verts: 
                        label = labels[i]
                        n_verts = temp
                        
                label.save('%s/rois/seed_%s_%s_%d-%dHz_%dTD_vs_%dASD_%s_cdt%s_pht%s-%s.label' 
                           % (paths['cluster'], roi_label, con_method, int(freqs[0]), 
                              int(freqs[-1]), n_TD, n_ASD, cond, cdt_p, pht_p, hemi))
                
        else:
            label = mne.read_label('%s/rois/%s' % (paths['cluster'], roi))
            label_verts = label.get_vertices_used(vertices=verts_fsave[0])
            cluster_mask = np.zeros(n_verts_fsave, dtype=bool)
            cluster_mask[np.searchsorted(verts_fsave[0], label_verts)] = True            
                        
                
        if do_corr and np.any(cluster_mask):
            for i_cond,cond in enumerate(conds):
                if cond not in corr_conds:
                    continue
                for score_label in score_labels:
                    if mode=='peak':
                        brains = data[:, i_cond, con_methods.index(con_method), 
                                      cluster_mask].max(-1)
                    elif mode=='mean':
                        brains = data[:, i_cond, con_methods.index(con_method), 
                                      cluster_mask].mean(-1)
                        
                    if save_brains:
                        np.save('%s/npy/seed%s_%s_N%d_%d-%dHz_%s_cdt%s_pht%s_%s_%s' % (paths['cluster'], 
                                                                            roi_label, con_method, 
                                                                            n_subs, fmin, fmax, cond, 
                                                                            cdt_p, pht_p, covar_out, 
                                                                            mode), brains)
                        
                    scores = np.array(sub_info[score_label])
                    diagnosed = sub_info['ASD'].copy()
                    
                    # indices of None, i.e. missing scores
                    null_inds = [i for i in range(len(scores)) 
                                 if scores[i]==None]
                    null_inds.reverse() # reverse so that largest index is removed first
                    # remove Nones and the corresponding brain data
                    for i in null_inds:
                        scores = np.delete(scores, i)
                        brains = np.delete(brains, i, axis=0)
                        diagnosed.pop(i)   
                        
                    this_inds_TD = [i for i in range(len(diagnosed)) if diagnosed[i]=='No']
                    this_inds_ASD = [i for i in range(len(diagnosed)) if diagnosed[i]=='Yes'] 
                           
                    fn_out = '%s/figures/connectivity/corr_seed%s_%s_%s_%s_cdt%s_pht%s' % (paths['cluster'], 
                                                                               roi_label, 
                                                                               con_method, cond, 
                                                                               score_label,
                                                                               cdt_p, pht_p)
    
                    if len(scores[this_inds_TD])>0:
                        r_TD, p_TD = pearsonr(scores[this_inds_TD], brains[this_inds_TD])
                        if p_TD<0.1:
                            print('\nTD: %s vs. %s for seed %s: r=%f, p=%f' 
                                  % (cond, score_label, roi_label, r_TD, p_TD))                    
    
                    if len(scores[this_inds_ASD])>0:
                        r_ASD, p_ASD = pearsonr(scores[this_inds_ASD], brains[this_inds_ASD])
                        if p_ASD<0.1:
                            print('\nASD: %s vs. %s for seed %s: r=%f, p=%f' 
                                  % (cond, score_label, roi_label, r_ASD, p_ASD))
                        
                    if len(scores[this_inds_TD])>0 and len(scores[this_inds_ASD])>0:
                        z,p = compare_corr_coefs(r_TD, r_ASD, len(this_inds_TD), len(this_inds_ASD))
                        if p<0.1:
                            print('\nTD vs. ASD: %s vs. %s for seed %s: z=%f, p=%f'
                                  % (cond, score_label, roi_label, z, p))
                        
                    if plot_corr and (p_ASD<0.05 or p_TD<0.05):
                        plt.figure()
                        if len(scores[this_inds_TD])>0:
                            ax = sns.regplot(list(scores[this_inds_TD]), 
                                             brains[this_inds_TD], ci=95, 
                                             scatter_kws={'s':80}, color='lightgreen', 
                                             marker='o', label='TD')
                            ax.text(0.025, 0.925, 'r=%.2f, p=%.3f' % (r_TD, p_TD), fontsize=12,                                    
                                    color='lightgreen', transform=ax.transAxes)
                        if len(scores[this_inds_ASD])>0:
                            ax = sns.regplot(list(scores[this_inds_ASD]), 
                                             brains[this_inds_ASD], ci=95, 
                                             scatter_kws={'s':80}, color='orchid', 
                                             marker='o', label='ASD')
                            ax.text(0.025, 0.875, 'r=%.2f, p=%.3f' % (r_ASD, p_ASD), fontsize=12,                                    
                                    color='orchid', transform=ax.transAxes)     
                            
                        if len(scores[this_inds_TD])>0 and len(scores[this_inds_ASD])>0:
                            ax.text(0.025, 0.825, 'z=%.2f, p=%.3f' % (z, p), fontsize=12,                                    
                                    color='k', transform=ax.transAxes) 
                        
                        ax.set_xlim(min(scores)-(max(scores)-min(scores))*0.05, 
                                    max(scores)+(max(scores)-min(scores))*0.05)
                        ax.set_ylim(min(brains)-(max(brains)-min(brains))*0.05, 
                                    max(brains)+(max(brains)-min(brains))*0.05)
                        plt.title('%s - %s seed connectivity within cluster correlation' 
                                  % (cond, roi_label))
                        plt.xlabel('%s' % score_label)
                        plt.ylabel('%s' % cond) 
                        ax.legend(loc='upper right')
                        plt.tight_layout()
                        plt.savefig('%s.png' % fn_out, dpi=300)
                        plt.close()
                        
        if mean_plot:
            group_labels = sub_info['ASD'] * len(plot_conds)
            # replace 'Yes' with 'ASD' and 'No' with 'TD' in group labels
            group_labels = ['ASD' if group_labels[i]=='Yes' else 'TD' 
                            for i in range(len(group_labels))]
            cond_labels = np.repeat(plot_conds, n_subs).tolist()
            
            inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
            inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
            
            this_data = {cond: data[:, conds.index(cond), con_methods.index(con_method), 
                                    cluster_mask].mean(-1) for cond in plot_conds}
            data_pd = {'Connectivity (%s)' % con_method: np.concatenate(list(this_data.values())), 
                       'Group': group_labels,
                       'Condition': cond_labels,
                       'ID': sub_info['sub_ID'] * len(plot_conds)}
            data_frame = pd.DataFrame(data=data_pd)
            
            c_TD = 'lightgreen'
            c_ASD = 'orchid'
            
            # plt.figure()
            if mean_plot=='bar':
                ax = sns.barplot(x='Condition', y='Connectivity (%s)' % con_method, hue='Group', 
                                 data=data_frame, alpha=1, edgecolor='gray', ci=68, 
                                 palette=[c_TD, c_ASD], hue_order=['TD', 'ASD'])
            elif mean_plot=='violin':
                if len(plot_conds)==1:
                    
                    fig, ax = plt.subplots()
                    ax = sns.violinplot(x='Group', y='Connectivity (%s)' % con_method, 
                                     data=data_frame, alpha=1, edgecolor='gray', 
                                     palette=[c_TD, c_ASD], order=['TD', 'ASD'],
                                     inner=None, cut=0)
                    # sns.swarmplot(x='Group', y='Connectivity (%s)' % con_method, 
                    #                 data=data_frame, size=10, alpha=1, 
                    #                 edgecolor='black', linewidth=1,
                    #                 palette=[c_TD, c_ASD], order=['TD', 'ASD'])
                    sns.stripplot(x='Group', y='Connectivity (%s)' % con_method, 
                                    data=data_frame, size=10, jitter=0.05, 
                                    alpha=1, edgecolor='black', linewidth=1,
                                    palette=[c_TD, c_ASD], order=['TD', 'ASD'])
                    
                    # figure size
                    # y_max = np.max(this_data[plot_conds[0]])
                    # y_min = np.min(this_data[plot_conds[0]])
                    y_max = 0.07
                    y_min = -0.07
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
                        ax.axvline(i, yminScaled, ymaxScaled, color='k', lw=sem_line_wid)
                    
                else:
                    ax = sns.violinplot(x='Condition', y='Connectivity (%s)' % con_method, 
                                         hue='Group', data=data_frame, alpha=1, edgecolor='gray', ci=68, 
                                         palette=[c_TD, c_ASD], hue_order=['TD', 'ASD'])
                    sns.stripplot(x='Group', y='Connectivity (%s)' % con_method, 
                                    data=data_frame, size=10, jitter=False, 
                                    alpha=1, edgecolor='black', linewidth=1,
                                    palette=[c_TD, c_ASD], order=['TD', 'ASD'])
            plt.title('seed A1 connectivity; %s %d-%dHz' % (con_method, fmin, fmax))
            plt.legend(loc='lower right')
            
            if mark_signifs:
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
                                
                signifs = []
                for val in p:
                    if val>0.06: signifs.append('')
                    elif val<0.06 and val>0.05: signifs.append('(*)')
                    elif val<0.05 and val>0.01: signifs.append('*')
                    elif val<0.01 and val>0.001: signifs.append('**')
                    elif val<0.001: signifs.append('***')
                    
                yloc = ax.get_ybound()[1] - (ax.get_ybound()[1] * 0.2)
                for signif, xloc in zip(signifs, ax.get_xticks()):
                    ax.text(xloc, yloc, signif, fontsize=20,
                            horizontalalignment='center') 
                    
            plt.tight_layout()
            fn_out = '%s/figures/connectivity/%s_seed%s_%s_%dTD_%dASD_%d-%dms_%d-%dHz_%s' \
                    % (paths['cluster'], mean_plot, roi_label, con_method, n_TD, n_ASD, 
                       int(tmin*1e3), int(tmax*1e3), fmin, fmax, 
                       ('_').join(plot_conds))
            if mark_signifs and nonparametric:
                    fn_out += '_ranksum'
            elif mark_signifs and not nonparametric:
                    fn_out += '_ttest'
            fn_out += '_%s_cdt%s_pht%s' % (covar_out, cdt_p, pht_p)
            plt.savefig('%s.png' % fn_out, dpi=300)
            plt.close()
                
#%% plot cluster time course
         
mask_name = 'seed_AC_wpli2_debiased_8-12Hz_28TD_vs_28ASD_MSS-SWS_cdt0.001_phtNone-lh'
cond = 'SWS'
con_method = 'wpli2_debiased'
tmin = -0.5
tmax = 1.5

linewidth = 3
c_TD = 'lightgreen'
c_ASD = 'orchid'

# load cluster mask
mask = mne.read_label('%s/rois/%s.label' % (paths['cluster'], mask_name))
if mask.hemi == 'lh':
    verts = verts_fsave[0]
else:
    verts = verts_fsave[1]
inds_mask = np.searchsorted(verts, mask.get_vertices_used(vertices=verts))

data_TD = data[inds_TD, conds.index(cond), con_methods.index(con_method)].mean(0)
data_ASD = data[inds_ASD, conds.index(cond), con_methods.index(con_method)].mean(0)

if tmin < times.min():
    i_tmin = 0
else:
    i_tmin = list(times).index(tmin)
if tmax > times.max():
    i_tmax = -1
else:
    i_tmax = list(times).index(tmax)+1

fig = plt.figure()
plt.plot(times[i_tmin:i_tmax], data_TD[inds_mask, i_tmin:i_tmax].mean(0), 
         label='TD', linewidth=linewidth, color=c_TD)
plt.plot(times[i_tmin:i_tmax], data_ASD[inds_mask, i_tmin:i_tmax].mean(0), 
         label='ASD', linewidth=linewidth, color=c_ASD)
plt.legend(loc='lower right')
plt.title('%s' % cond)
plt.xlabel('Time (s)')
plt.ylabel('Connectivity (%s)' % (' ').join(con_method.split('_')))
plt.tight_layout()
fig.savefig('%s/figures/connectivity/tcs_%dTD_%dASD_%s_%s_%s_%d-%dms_%d-%dHz.png' 
            % (paths['cluster'], n_TD, n_ASD, mask_name, cond, con_method, 
               int(tmin*1e3), int(tmax*1e3), int(fmin), int(fmax)), dpi=300)
plt.close()

#%% save individual ROIs

import os
import subprocess

mask_name = 'seed_AC_wpli2_debiased_8-12Hz_28TD_vs_28ASD_MSS-SWS_cdt0.01_phtNone-lh'
cond = 'MSS-SWS'
con_method = 'wpli2_debiased'
tmin = 0.
tmax = 1.5
n_verts = 5
peak = False
adjacent = False

for i_sub,sub_ID in enumerate(sub_IDs):
    print(sub_ID)
    sub_path = '%s/%s' % (paths['cluster'], sub_ID)
    
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                                 % (sub_path, sub_ID))
    src = inv['src']
    verts = [s['vertno'] for s in src]
        
    data_sub = data[i_sub, conds.index(cond), con_methods.index(con_method),
                    :, list(times).index(tmin):list(times).index(tmax)+1]
    stc = mne.SourceEstimate(data_sub, vertices=verts_fsave, tmin=tmin, 
                             tstep=np.round(times[1]-times[0], decimals=3), 
                             subject='fsaverageJA')
    mask = mne.read_label('%s/rois/%s.label' % (paths['cluster'], mask_name))
    if peak:
        _,i_time = stc.in_label(mask).get_peak(hemi=mask.hemi, tmin=tmin, 
                                               tmax=tmax, mode='pos',
                                               time_as_index=True)
    else:
        stc = stc.mean()
        i_time = 0
        
    brain = np.zeros(stc.data.shape[0])
    if mask.hemi=='lh':
        inds_verts = np.searchsorted(verts_fsave[0], 
                                     mask.get_vertices_used(verts_fsave[0]))
        brain[inds_verts] = stc.copy().data[inds_verts, i_time]
        brain[len(verts_fsave[0])::] = 0
    else:
        inds_verts = len(verts[0]) + \
            np.searchsorted(verts_fsave[1], mask.get_vertices_used(verts_fsave[1]))
        brain[inds_verts] = stc.copy().data[inds_verts, i_time]
        brain[0:len(verts_fsave[1])] = 0
        
    # threshold of highest n_verts values
    th = np.sort(np.squeeze(brain))[::-1][n_verts]
    # indices of top n_verts vertices
    inds = np.where(brain>th)
    
    # make stc with the thresholded data
    brain_th = np.zeros(brain.shape)
    brain_th[inds] = brain[inds]
    stc_th = mne.SourceEstimate(brain_th, verts_fsave, tmin, tstep=0, 
                                subject='fsaverageJA')
        
    # make ROI from stc
    if adjacent:
        froi = mne.stc_to_label(stc_th, src=src_fsave, smooth=True, connected=True,
                                subjects_dir=paths['fs'])[0]
        froi = froi[0]
    else:
        froi = mne.stc_to_label(stc_th, src=src_fsave, smooth=True,
                               subjects_dir=paths['fs'])
        froi = list(filter(None, froi))[0]
        
    # morph label to the individual
    froi_morphed = froi.morph(subject_from='fsaverageJA', subject_to=fs_id,
                              smooth=5, grade=verts, subjects_dir=paths['fs'])
        
    if peak:
        froi_name = 'A1seedCon_%s_%dverts_%s_peak%d-%dms' % (con_method, n_verts, cond,
                                                        int(tmin*1000), int(tmax*1000))
    else:
        froi_name = 'A1seedCon_%s_%dverts_%s_mean%d-%dms' % (con_method, n_verts, cond,
                                                        int(tmin*1000), int(tmax*1000))
    if adjacent:
        froi_name += '_adjVerts'

        
    # save ROI png
    out_folder = '%s/figures/misc/rois/%s/' % (paths['cluster'], froi_name)
    if not os.path.exists(out_folder):
        subprocess.call('mkdir %s' % out_folder, shell=True)
    fig = stc_th.plot(subject='fsaverageJA', subjects_dir=paths['fs'], hemi=mask.hemi, 
                        backend='matplotlib', spacing='ico5')
    fig.savefig('%s/%s_%s-%s.png' % (out_folder, sub_ID, froi_name, mask.hemi), dpi=300)
    plt.close()
    # save ROI
    froi.save('%s/rois/%s' % (sub_path, froi_name))
