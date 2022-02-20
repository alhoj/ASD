
import numpy as np
import mne
import pickle
from matplotlib import pyplot as plt

roi_names = [
              # 'AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-lh'
              'AC_dSPM_5verts_MSS_SWS_peak0-500ms_auto-rh'
            ]

overlap_type = 'prob' # 'prob' or 'nsubs'
lims = [0.3, 0.5, 0.8] # plotting limits in probability
bg = 'w'

#%%

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/'
         }

# read subject IDs and their functional ROIs labels
# read subject info
f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

# exclude = ['105801', '107301']
exclude = ['105801', '107301', '052902', '090902', '048102', '075401', '096603']
for sub in exclude:
    ind = sub_info['sub_ID'].index(sub)
    n_subs = len(sub_info['sub_ID'])
    [sub_info[key].pop(ind) for key in sub_info.keys() 
     if len(sub_info[key])==n_subs]
sub_IDs = sub_info['sub_ID']
n_subs = len(sub_IDs)
inds_ASD = [i for i in range(n_subs) if sub_info['ASD'][i]=='Yes']
inds_TD = [i for i in range(n_subs) if sub_info['ASD'][i]=='No']
n_ASD = len(inds_ASD)
n_TD = len(inds_TD)

# read the source space we are morphing to
src_fname = '%s/fsaverageJA/bem/fsaverageJA-oct6-src.fif' % paths['fs']
src = mne.read_source_spaces(src_fname)
adjacency = mne.spatial_src_adjacency(src)
verts_fsave = [s['vertno'] for s in src]
n_verts_fsave = len(np.concatenate(verts_fsave))

if overlap_type == 'prob':
    title = 'Probability'
elif overlap_type == 'nsubs':
    title = 'Number of subjects'
    lims = [lims[0]*n_subs, lims[1]*n_subs, lims[2]*n_subs]
    
#%%
for roi_name in roi_names:
    rois = []
    for sub_ID in sub_IDs: # loop subject by subject 
        print(sub_ID)
        sub_path = '%s/%s/' % (paths['cluster'], sub_ID)
        fs_id = sub_info['FS_dir'][sub_info['sub_ID'].index(sub_ID)]
        # read subjects' corresponding functional labels
        roi = mne.read_label('%s/rois/%s.label' % (sub_path, roi_name))
        roi.subject = fs_id
        # morph the subjects' functional labels to fsaverage, and append each 
        # morphed label to the holder variable
        rois.append(roi.morph(subject_to='fsaverageJA', grade=verts_fsave))
        
#%         
    hemi = rois[0].hemi
    data = np.zeros(n_verts_fsave)
    for roi in rois:
        if hemi=='lh':
            inds = np.searchsorted(verts_fsave[0], roi.vertices)
        else:
            inds = len(verts_fsave[0]) + np.searchsorted(verts_fsave[1], roi.vertices)
        data[inds] += 1
    if overlap_type == 'prob':
        data /= n_subs
    
    # create stc
    stc = mne.SourceEstimate(data, subject='fsaverageJA',
                             vertices=verts_fsave, tmin=0, tstep=0)
    
    brain = stc.plot(subject='fsaverageJA', subjects_dir=paths['fs'], 
                    hemi=hemi, backend='matplotlib', spacing='ico7',
                    background=bg, cortex='low_contrast',  colorbar=True, 
                    colormap='jet', clim=dict(kind='value', lims=lims))
    # plt.title('Number of subjects', color='w')
    plt.title(title, color='w')
    brain.savefig('%s/figures/overlap_%dTD_%dASD_%s_%s_%s.png' % (paths['cluster'], n_TD, n_ASD,
                                                                  roi_name, overlap_type, bg), dpi=500)
    plt.close()
    

