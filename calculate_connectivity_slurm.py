#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:00:25 2020

@author: ja151
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:51:58 2020

@author: ja151
"""

import sys
import mne
import pickle
import numpy as np
import os

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

#%% subject ID from command line argument
sub_ID = sys.argv[1]
print('Calculating connectivity for subject %s' % sub_ID, flush=True)

#%%

# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# conditions
conds = ['MSS', 'SWS']
n_conds = len(conds)
equalize_event_counts = True
add_baseline_cond = False

# times
tmin = -0.5 # None for beginning of epochs
tmax = 1.5 # None for end of epochs
average_times = False

# freqs
fmin = 8
fmax = 12
fstep = 0.5
freqs = np.arange(fmin, fmax+fstep, fstep)
n_freqs = len(freqs)
n_cycles = 7.
average_freqs = True

# connectivity params
con_methods = ['wpli2_debiased']
con_mode = 'cwt_morlet'

# rois
roi_names = ['AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh'
             # 'AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-rh'
             # 'AC_MNE_5verts_MSS_SWS_peak0-200ms_auto-lh',
             # 'AC_MNE_10verts_MSS_SWS_peak0-200ms_auto-lh']
             # 'IP_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh',
             # 'pac_MSS-SWS_TD_vs_ASD_cluster_phase9.5-10.5Hz_amp40-50Hz-lh',
             # 'ACseedConn_TDvsASD_cluster-lh'
             ]
# roi_mask_names = ['LOBE.FRONTAL', 'LOBE.TEMPORAL', 'LOBE.PARIETAL']
rois_as_seed = True # calculate connectivity from the ROIs to the rest of the brain
                    # or between the ROIs only

# params for source estimates
snr = 1.0
lambda2 = 1.0 / snr ** 2
stc_method = 'MNE'
pick_ori = 'normal'
tc_extract = 'mean_flip'

out_id = ''
overwrite = True
n_jobs = 1

# read the source space we are morphing to
src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
src = mne.read_source_spaces(src_fname)
verts_fsave = [s['vertno'] for s in src]
n_verts_fsave = len(np.concatenate(verts_fsave))

#%% 

sub_path = '%s/%s/' % (paths['cluster'], sub_ID)    
inv = mne.minimum_norm.read_inverse_operator('%s/%s_1-100Hz_notch60Hz-oct6-inv.fif' 
                                             % (sub_path, sub_ID))
src = inv['src']    
verts = [s['vertno'] for s in src]
fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
epochs = mne.read_epochs('%s/%s_speech_1-100Hz_notch60Hz_-500-2000ms-epo.fif' 
                         % (sub_path, sub_ID), proj=False)
sfreq = epochs.info['sfreq']
times = epochs.times

if tmin is not None: 
    i_tmin = list(times).index(tmin)
else: 
    i_tmin = 0
    tmin = times[i_tmin]
if tmax is not None: 
    i_tmax = list(times).index(tmax)
else: 
    i_tmax = -1
    tmax = round(times[i_tmax])

if equalize_event_counts:
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

if add_baseline_cond: 
    # epoch_length = tmax-tmin
    # baseline_length = abs(times[0])
    # multiplier = int(epoch_length / baseline_length)
    # n_epochs_bl = n_epochs*multiplier
    epochs_bl = epochs[0:n_epochs]
    i_tmin_bl = 0
    i_tmax_bl = list(times).index(0)
    conds += ['Baseline']

rois = []
for roi_name in roi_names:
    roi_path = '%s/%s/rois/%s.label' % (paths['cluster'], sub_ID, 
                                        roi_name)
    roi_path_fsave = '%s/rois/%s.label' % (paths['cluster'], roi_name)
    if os.path.exists(roi_path):
        rois.append(mne.read_label(roi_path, subject=fs_id))
    elif os.path.exists(roi_path_fsave):
        roi = mne.read_label(roi_path_fsave, subject='fsaverageJA')
        rois.append(roi.copy().morph(subject_to=fs_id, 
                                     subject_from='fsaverageJA',
                                     grade=verts,
                                     subjects_dir=paths['fs']))
    else:
        rois.append(mne.read_labels_from_annot('fsaverageJA', 
                                                parc='PALS_B12_Lobes', 
                                                subjects_dir=paths['fs'], 
                                                regexp=roi_name)[0])


# rois_mask = []
# for roi_mask_name in roi_mask_names:
#     roi_path = '%s/%s/rois/%s-%s.label' % (paths['cluster'], sub_ID, 
#                                            roi_mask_name, hemi)
#     if os.path.exists(roi_path):
#         rois_mask.append(mne.read_label(roi_path, subject=fs_id))            
#     else:
#         rois_mask.append(mne.read_labels_from_annot('fsaverageJA', 
#                                                     parc='PALS_B12_Lobes', 
#                                                     subjects_dir=paths['fs'], 
#                                                     regexp=roi_mask_name)[0])
# if fsaverage ROIs, morph ROIs to subject
   
# if rois_mask[0].subject == 'fsaverageJA':
#     temp = []
#     for roi_mask in rois_mask:
#         temp.append(roi_mask.copy().morph(subject_to=fs_id, 
#                                           subject_from='fsaverageJA',
#                                           subjects_dir=paths['fs']))
        
#     rois_mask = temp
    
# # all ROIs mask
# mask = rois_mask[0]
# for roi_mask in rois_mask[1::]: mask += roi_mask

#%%
if rois_as_seed:
    for roi in rois:
        print('\nCalculating connectivity from %s to the rest of the brain\n' % roi.name)
        roi_label = roi.name#'%s-%s' % (roi.name[0:2], roi.name[-2:])
        # output filename
        fn_out = 'con_%d-%dms_%d-%dHz_%s' % (int(tmin*1000), int(tmax*1000), 
                                             fmin, fmax, '-'.join(conds))
        fn_out = fn_out.replace('/','')
        fn_out += '_seedROI_%s' % roi_label
        fn_out += '_%s' % '-'.join(con_methods)
        if out_id: fn_out += '_%s' % out_id
        out_path = '%s/p/%s.p' % (sub_path, fn_out)    
        # check if output already exist
        if os.path.exists(out_path) and not overwrite:
            print('Output already exists for %s' % sub_ID)
            continue
        
        cons_dict = {}
        for cond in conds:
            print('\nCondition: %s' % cond, flush=True)
            if cond=='Baseline':
                this_epochs = epochs_bl
            else:
                this_epochs = epochs[cond]
            stcs = mne.minimum_norm.apply_inverse_epochs(this_epochs, 
                                                         inverse_operator=inv, 
                                                         lambda2=lambda2, 
                                                         method=stc_method, 
                                                         pick_ori=pick_ori,
                                                         return_generator=False)
            
            roi_tcs = mne.extract_label_time_course(stcs, [roi], src, tc_extract,
                                                    return_generator=False)

            tcs = list(zip(roi_tcs, stcs))
            n_signals = 1 + len(np.concatenate(verts))
            inds = mne.connectivity.seed_target_indices([0], np.arange(1, n_signals))
                
            con, freqs, _, n_epochs, _ = mne.connectivity.spectral_connectivity(
                tcs, method=con_methods, mode=con_mode, indices=inds, 
                sfreq=sfreq, cwt_freqs=freqs, cwt_n_cycles=n_cycles, n_jobs=n_jobs)
            
            if average_times:
                if cond=='Baseline':
                    i_tmin = i_tmin_bl
                    i_tmax = i_tmax_bl
                if type(con)==list:
                    con = [this_con[:,:,i_tmin:i_tmax].mean(-1) for this_con in con]
                else:
                    con = con[:,:,i_tmin:i_tmax].mean(-1)
        
            if average_freqs:
                if type(con)==list:
                    con = [this_con.mean(1) for this_con in con]
                else:
                    con = con.mean(1)
            
            cons_dict.update({cond: con})
            # n_epochs_dict.update({cond: n_epochs})
        
        data_struct = {'cons': cons_dict, 'conds': conds, 'rois': roi.name, 
                       'times': (tmin, tmax), 'freqs': freqs, 'n_epochs': n_epochs,
                       'n_cycles': n_cycles, 'stc_method': stc_method, 
                       'lambda2': lambda2, 'pick_ori': pick_ori, 
                       'con_method': con_methods, 'con_mode': con_mode,
                       'rois_as_seed': rois_as_seed, 'average_times': average_times,
                       'average_freqs': average_freqs}
        
        #% save connectivity data
        print('Saving %s' % out_path)
        f = open(out_path, 'wb')
        pickle.dump(data_struct, f)
        f.close()
        
else:
    # output filename
    fn_out = 'con_%d-%dms_%d-%dHz_%s' % (int(tmin*1000), int(tmax*1000), 
                                            fmin, fmax, '-'.join(conds))
    fn_out = fn_out.replace('/','')
    fn_out += '_betweenROIs'    
    fn_out += '_%s' % '-'.join(con_methods)
    out_path = '%s/p/%s.p' % (sub_path, fn_out)    
    # check if output already exist
    if os.path.exists(out_path) and not overwrite:
        print('Output already exists for %s' % sub_ID)
        exit()
    
    cons_dict = {}
    for cond in conds:
        print('\nCondition: %s' % cond, flush=True)
        if cond=='Baseline':
            this_epochs = epochs_bl
        else:
            this_epochs = epochs[cond]
        stcs = mne.minimum_norm.apply_inverse_epochs(this_epochs, 
                                                     inverse_operator=inv, 
                                                     lambda2=lambda2, 
                                                     method=stc_method, 
                                                     pick_ori=pick_ori,
                                                     return_generator=False)
        
        roi_tcs = mne.extract_label_time_course(stcs, rois, src, tc_extract,
                                                return_generator=False)
           
        con, freqs, _, n_epochs, _ = mne.connectivity.spectral_connectivity(
            roi_tcs, method=con_methods, mode=con_mode, indices=None, 
            sfreq=sfreq, cwt_freqs=freqs, cwt_n_cycles=n_cycles, n_jobs=n_jobs)
        
        if average_times:
            if type(con)==list:
                con = [this_con[:,:,:,i_tmin:i_tmax].mean(-1) for this_con in con]
            else:
                con = con[:,:,:,i_tmin:i_tmax].mean(-1)
    
        if average_freqs:
            if type(con)==list:
                con = [this_con.mean(-1) for this_con in con]
            else:
                con = con.mean(1)
        
        cons_dict.update({cond: con})
        # n_epochs_dict.update({cond: n_epochs})
    
    data_struct = {'cons': cons_dict, 'conds': conds, 'rois': roi_names, 
                   'times': (tmin, tmax), 'freqs': freqs, 'n_epochs': n_epochs,
                   'n_cycles': n_cycles, 'stc_method': stc_method, 
                   'lambda2': lambda2, 'pick_ori': pick_ori, 
                   'con_method': con_methods, 'con_mode': con_mode,
                   'rois_as_seed': rois_as_seed, 'average_times': average_times,
                   'average_freqs': average_freqs}
    
    #% save connectivity data
    print('Saving %s' % out_path)
    f = open(out_path, 'wb')
    pickle.dump(data_struct, f)
    f.close()
                
