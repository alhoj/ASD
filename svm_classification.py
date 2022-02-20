#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:13:18 2020

@author: ja151
"""
import mne
from MNEprepro import MNEprepro
import pickle
import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_validate
import scipy
import matplotlib.pyplot as plt
import os

paths = {'cluster': '/autofs/cluster/transcend/jussi/',
         'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# define bad subjects
bad_subs = ['076301', '100001', '052901', '096603', '032902', '013703', '011201', 
            '048102', '052902', '075401']
bad_sub_inds = [sub_info['sub_ID'].index(bad_sub) for bad_sub in bad_subs]
bad_sub_inds.sort()

n_subs = len(sub_info['sub_ID'])
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
n_TD = len(inds_TD)
n_ASD = len(inds_ASD)

conds = ['Speech', 'Jabber']
n_conds = len(conds)

grade = 5
src_fname = paths['fs'] + '/fsaverageJA/bem/fsaverageJA-ico%d-src.fif' % grade
src = mne.read_source_spaces(src_fname)
fsave_vertices = [s['vertno'] for s in src]
n_vert_fsave = len(fsave_vertices[0])+len(fsave_vertices[1])

# Parameters for source estimate
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'MNE'

# ROIs
roi_labels = ['aSTG', 'aMTG', 'mSTG', 'mMTG', 'IPS']
rois = []
for roi_ind,roi_label in enumerate(roi_labels):
    rois.append(mne.read_label('%s/rois/%s-lh.label' % (paths['out'], 
                                                        roi_labels[roi_ind])))
    
n_rois = len(rois)
# assign the name of the subject
for roi in rois: roi.subject='fsaverageJA'

#%% Read timecourses from fROIs

use_fsaverage_rois = False # if True, morph individual source estimates to fsaverage; 
                          # if False, morph ROIs to individual space 
tmin = 0
tstep = 0.001
tmax = 2.0
times = np.arange(tmin,tmax,tstep)
n_times = int(len(times))

tcs = np.zeros((n_subs, n_conds, n_rois, n_times))
targets = []

for i_sub,sub_ID in enumerate(sub_info['sub_ID']):
    sub = MNEprepro(sub_ID,paths)
    sub.get_epochs(postfix='_0-40Hz_-200-2000ms_noReject')
    picks = mne.pick_types(sub.epochs.info, meg=True)
    sub.make_inv_operator()    
    fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
    
    # targets/labels for classification: ASD -> 1, TD -> 0
    if sub_info['ASD'][sub_info['sub_ID'].index(sub_ID)]=='Yes': 
        targets.append(1)
    else: 
        targets.append(0) 
    
    for i_cond,cond in enumerate(conds):                                            
        evoked = sub.epochs[cond].apply_baseline().average(picks=picks)
        stc = mne.minimum_norm.apply_inverse(evoked, sub.inv, lambda2, method)
        stc.crop(tmin=tmin, tmax=tmax, include_tmax=False)
                
        if use_fsaverage_rois: # morph individual source estimate to freesurfer average brain
            morph = mne.compute_source_morph(stc, fs_id, 'fsaverageJA',
                     spacing=fsave_vertices, subjects_dir=paths['fs'])        
            stc = morph.apply(stc)  
            src = mne.read_source_spaces(src_fname)

        for i_roi,roi in enumerate(rois):
            if not use_fsaverage_rois: # morph fsaverage ROIs to individual space
                src = sub.inv['src'] # get the source space
                roi = roi.copy().morph(subject_to=fs_id, subject_from='fsaverageJA',
                                       subjects_dir=paths['fs'])
            
            # extract timecourse from ROI
            tcs[i_sub, i_cond, i_roi] = stc.extract_label_time_course(roi, src, 
                                                                     mode='pca_flip')[0]
    
tcs_orig = np.abs(tcs)
targets_orig = targets
bad_sub_inds_orig = bad_sub_inds

#%% run classification

n_fold = 3 # 1 for leave-one-out
cond = 'Speech-Jabber'
equal_samples = True # use equal sample sizes (NVIQ-matched)

#Regress covariates out of the data
regress_covars = True
covar_names = ['age', 'NVIQ', 'VIQ']

if equal_samples:
    bad_sub_inds = bad_sub_inds_orig
    tcs = np.delete(tcs_orig, bad_sub_inds, axis=0)
    targets = np.delete(targets_orig, bad_sub_inds)
else:
    tcs = tcs_orig
    targets = targets_orig
    bad_sub_inds = []
n_subs = tcs.shape[0]

if regress_covars:    
    print('Regressing out covariates: %s' % covar_names)    
    covars = []
    for covar_name in covar_names:
        covars.append([val for ind,val in enumerate(sub_info[covar_name]) 
                       if ind not in bad_sub_inds])
    
    # make model out of covariates
    model = np.zeros((n_subs, len(covar_names)+1))
    model[:,0] = np.ones(n_subs)
    model[:,1::] = np.transpose(covars)
     
    # reshape time course data
    tcs = np.reshape(tcs, (n_subs, n_conds*n_rois*n_times))

    # get beta coefficients
    beta = scipy.linalg.lstsq(model, tcs)[0]
    
    # regress covariates out from time courses
    tcs = tcs - model.dot(beta)
    
    # reshape back to original shape
    tcs = np.reshape(tcs, (n_subs, n_conds, n_rois, n_times))
    
else: 
    print('No covariates')
    covar_names = ['nothing']

# condition
if cond == 'Speech':
    tcs = tcs[:,0]
elif cond == 'Jabber':
    tcs = tcs[:,1]
elif cond == 'Speech-Jabber':
    tcs = tcs[:,0]-tcs[:,1] # Speech-Jabber contrast
    # tcs_mean = tcs_mean[:,0]/tcs_mean[:,1] # Speech/Jabber ratio
elif cond == 'Speech&Jabber':
    tcs = tcs.reshape(n_subs, n_conds*n_rois, n_times)

#% average time courses within time windows of interest
tmin_clusters = [0.84, 0.97, 1.13, 1.00, 1.05]
tmax_clusters = [1.48, 1.48, 1.46, 1.54, 1.32]

tcs_mean = np.zeros((n_subs, n_rois))
for i_roi in range(n_rois):
    # cluster temporal extent indices
    i_tmin_roi = int(tmin_clusters[i_roi]*1000-1)
    i_tmax_roi = int(tmax_clusters[i_roi]*1000-1)
    tcs_mean[:,i_roi] = tcs[:,i_roi,i_tmin_roi:i_tmax_roi].mean(-1)

#% svm classification with k-fold cross-validation
y = np.array(targets)
X = tcs_mean
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
# X = scipy.stats.zscore(X)

if n_fold>1:
    clf = svm.LinearSVC(dual=False)
    cv = RepeatedKFold(n_splits=n_fold, n_repeats=1000, random_state=1)
    scoring = ['balanced_accuracy', 'recall']
    scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, n_jobs=12)
    balanced_acc = np.mean(scores['test_balanced_accuracy'])
    sensitivity = np.mean(scores['test_recall'])
    specificity = 2*balanced_acc-sensitivity
else: # leave-one-out cross validation
    clf = svm.SVC(kernel='linear')
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    y_pred = np.zeros(y.shape)
    for train_ind, test_ind in loo.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        clf.fit(X_train, y_train) # train the classifier
        y_pred[test_ind] = clf.predict(X_test)

# accuracy = accuracy_score(y, y_pred)
# fpr, tpr,_ = roc_curve(y, y_pred, pos_label=1)
# sensitivity = tpr[1]
# specificity = 1 - fpr[1]
# balanced_acc = (sensitivity+specificity)/2

print('Balanced accuracy: %f; sensitivity: %f; specificity: %f' % (balanced_acc, 
                                                                   sensitivity,
                                                                   specificity))


#%% svm classification with k-fold cross-validation

n_fold = 6
cond = 'Speech-Jabber'

if cond == 'Speech':
    tcs = tcs[:,0,:,:]
elif cond == 'Jabber':
    tcs = tcs[:,1,:,:]
elif cond == 'Speech-Jabber':
    tcs = tcs[:,0,:,:]-tcs[:,1,:,:] # Speech-Jabber contrast
#data = data_full[:,0,:,:]/data_full[:,1,:,:] # Speech/Jabber ratio

y = np.array(targets)
clfs = []
accuracies = []
for time in np.arange(n_times):
    print(time)
    X = tcs[:,:,time]
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    # X = scipy.stats.zscore(X)
    
    clf = svm.LinearSVC(C=1.0, random_state=0, max_iter=1000)
    rkf = RepeatedKFold(n_splits=n_fold, n_repeats=1)
    
    y_pred = np.zeros(y.shape)
    for train_ind, test_ind in rkf.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        clf.fit(X_train, y_train) # train the classifier
        y_pred[test_ind] = clf.predict(X_test)
    accuracies.append(metrics.accuracy_score(y, y_pred))
    clfs.append(clf)
  
cv_label = str(n_fold) + 'fold'

#%% svm classification with leave-one-out cross-validation

#data = data_full[:,1,:,:]
data = data_abs[:,0,:,:]-data_abs[:,1,:,:] # Speech-Jabber contrast
#data = data_full[:,0,:,:]/data_full[:,1,:,:] # Speech/Jabber ratio

y = np.array(targets)
clfs = []
accuracies = []
for time in range(n_time):
    print(time)
    X = data[:,:,time]
    X = scipy.stats.zscore(X)
    
    clf = svm.SVC(kernel='linear')
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    y_pred = []
    for train_ind, test_ind in loo.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        clf.fit(X_train, y_train) # train the classifier
        y_pred.append(clf.predict(X_test.reshape(1,-1))[0])
    accuracies.append(metrics.accuracy_score(y, y_pred))
    clfs.append(clf)
  
cv_label = 'loo'  


#%% plot accuracies

plt.plot(accuracies)
name_png = '%s/figures/classifAcc_%s_%dASD-%dTD_%s_7ROIaparc_grade%d_fs%dHz_%sRegressed.png' \
        % (paths['out'], cv_label, n_ASD, n_TD, data_label, grade, fs_resample, ''.join(covar_labels))
if mov_ave: name_png = '%s_movAve%d.png' % (os.path.splitext(name_png)[0], mov_ave) 
plt.savefig(name_png, dpi=300)
plt.close()

#%% save stc

#accuracies_sorted = sorted(accuracies,reverse=True)
#ind = accuracies.index(accuracies_sorted[1])

# replace Nones with zeros
accuracies  = [0 if accuracy==None else accuracy for accuracy in accuracies] 
ind = accuracies.index(np.max(accuracies))

weights = np.zeros(n_vert_fsave)
weights[mask.get_vertices_used()] = clfs[ind].coef_
stc_weights = mne.SourceEstimate(weights, vertices=fsave_vertices, 
                               tmin=0, tstep=0.001, subject='fsaverageJA')
name_stc = '%s/SVMweights_%s_%dASD-%dTD_%s_7ROIaparc_grade%d_fs%dHz_%sRegressed_%dms_acc%d' \
        % (paths['out'], cv_label, n_ASD, n_TD, data_label, grade, fs_resample, ''.join(covar_labels), 
           int(ind*int(fs/fs_resample)), int(round(accuracies[ind]*100)))
if mov_ave: name_stc = '%s_movAve%d' % (os.path.splitext(name_stc)[0], mov_ave) 
stc_weights.save(name_stc)

#%%
ind_min = 94
ind_max = 131
temp = np.zeros((n_vert,ind_max-ind_min))
for i,ind_clf in enumerate(np.arange(ind_min,ind_max)):
    temp[:,i] = clfs[ind_clf].coef_
temp = np.mean(temp,axis=1)

weights = np.zeros(n_vert_fsave)
weights[mask.get_vertices_used()] = temp
stc_weights = mne.SourceEstimate(weights, vertices=fsave_vertices, 
                               tmin=0, tstep=0.001, subject='fsaverageJA')

stc_weights.save('%s/svm_weights_Speech-Jabber_%s_%sRegressed_%s_mean%d-%dms' 
                 % (paths['out'], roi, ''.join(covar_labels), cv_label, 
                    int(ind_min*10), int(ind_max*10)))