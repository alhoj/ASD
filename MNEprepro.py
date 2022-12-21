#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:31:59 2019

@author: ja151
"""

import os
import mne
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from itertools import compress
import pickle
import subprocess

print(__doc__)


paths = {'speech': '/autofs/cluster/transcend/MEG/speech/',
         'local': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/',
         'cluster': '/autofs/cluster/transcend/jussi/',
         'cb': '/autofs/cluster/transcend/data_exchange/cerebellar_source_spaces_autism_cohort/source_spaces'
         }

f = open('%s/p/subjects.p' % paths['cluster'], 'rb')
sub_info = pickle.load(f)
f.close()

class MNEprepro():
    """
    Container for subject-specific paths, names, and preprocessing functions
    
    Parameters
    ----------
    sub : str
        Subject ID
        
    Attributes
    ----------
    erm_name : list
        Name(s) of the empty room recording fif file(s)
    fs_id_new : str
        FreeSurfer ID (updated)
    fs_id_old : str
        FreeSurfer ID (original)
    path_cluster : str
        Subject directory in /autofs/cluster/
    path_erm : str
        Empty room MEG directory
    path_fs : str
        FreeSurfer reconstruction directory
    path_local : str
        Subject directory in /local_mount/space/hypatia/
    path_speech : str
        Speech paradigm MEG directory
    raw_names : list
        Name(s) of the Speech paradigm raw fif file(s)
    src_cb : str
        Cerebellar source space -src.fif file
    sub : str
        Subject ID
        
    """
    def __init__(self, sub):        
        self.sub = sub
        self.fs_id_new = sub_info['FS_dir_new'][sub_info['sub_ID'].index(self.sub)]
        self.fs_id_old = sub_info['FS_dir'][sub_info['sub_ID'].index(self.sub)]
        self.path_fs = os.path.join(paths['fs'], self.fs_id_new)
        self.path_local = os.path.join(paths['local'], self.sub)
        self.path_cluster = os.path.join(paths['cluster'], self.sub)
        self.src_cb = '%s/%s-src.fif' % (paths['cb'], self.fs_id_old)
        
        # check if original raw-files exist in local directory (or if the directory even exists)
        if not os.path.exists(self.path_local):
            subprocess.call('mkdir %s' % self.path_local, shell=True)
            self.raw_names = []
        else:
            self.raw_names = [file for file in os.listdir(self.path_local) 
                              if file.endswith('raw.fif') and 'speech' in file]
            self.erm_name = [file for file in os.listdir(self.path_local) 
                             if file.endswith('raw.fif') and 'erm' in file]
        
        # otherwise read file names from the original speech directory
        if not self.raw_names:
            session = os.listdir(os.path.join(paths['speech'], self.sub))
            if len(session) > 2: session='/'
            elif len(session) == 2: session=session[0]
            self.path_speech = os.path.join(paths['speech'], self.sub, session[0])             
            self.path_erm = os.path.join(paths['erm'], self.sub, session[0])            
            self.raw_names = [file for file in os.listdir(self.path_speech) 
                              if file.endswith('raw.fif') and 'speech' in file]
            self.erm_name = [file for file in os.listdir(self.path_erm) 
                             if file.endswith('raw.fif') and 'erm' in file]
        else:
            self.path_speech = self.path_local
            self.path_erm = self.path_local
            
    
    def detect_bad_channels(self, method='power_log', zscore_th=6.0, neigh_max_dist=0.1, plot=True, overwrite=False):                        
        """
        Detect bad channels from raw fif file
        
        Parameters
        ----------
        method : str
            Method to use. Options are 'corr' (local uncorrelation), 'amplitude' (amplitude range),
            'power' (magnitude), 'corr_power' (combine uncorrelation with magnitude),
            'amplitude_log' (amplitude + log amplitude), 'power_log' (default; power + log power)
        zscore_th : float
            Threshold for detecting bad channel (default 6) 
        neigh_max_dist : float
            Maximum distance of neighboring channels (applies only if method is 'corr' or 'corr_power')
        plot : bool
            Plot channels and diagnostics (default True)
        overwrite : bool
            Overwrite if already run (default False)
        """
        print('Detecting bad channels...')
        
        self.raw = [] # initialize list
        for raw_name in self.raw_names:   
            
            raw = mne.io.read_raw_fif(os.path.join(self.path_speech, raw_name), preload=False)
            out_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0] + '_bads.txt')   
            os.makedirs(self.path_local, exist_ok=True)
                            
            if os.path.exists(out_fname) and not overwrite:
                bad_chns = open(out_fname, "r").read().splitlines()
                raw.info['bads'] = bad_chns
                if bad_chns:
                    print('Reading from file, bad channels:', bad_chns)
                else:
                    print('Reading from file, no bad channels')
            else:                        
                # Use only a short, filtered, and downsampled segment
                fs = raw.info['sfreq']
                t1x = 100
                t2x = 200
                t2 = int(round(min(raw.last_samp/fs, t2x)))
                t1 = int(round(max(0, t1x + t2-t2x)))  # Start earlier if recording is shorter
                f1 = 1 # highpass freq
                f2 = 40 # lowpass freq
                Fs_resample = 100
        
                # Get data
                raw_copy = raw.copy().load_data().pick_types(meg=True).filter(f1, f2).resample(Fs_resample, npad='auto')
                data = raw_copy.get_data(start=t1*Fs_resample, stop=t2*Fs_resample)
        
                # Get channel distances matrix
                chns_locs = np.asarray([x['loc'][:3] for x in raw_copy.info['chs']])
                chns_dist = np.linalg.norm(chns_locs - chns_locs[:, None], axis=-1)
                chns_dist[chns_dist > neigh_max_dist] = 0
        
                # Get avg channel uncorrelation between neighbours
                chns_corr = np.abs(np.corrcoef(data))
                weights = np.array(chns_dist, dtype=bool)
                chn_nei_corr = np.average(chns_corr, axis=1, weights=weights)
                chn_nei_uncorr_z = zscore(1-chn_nei_corr)  # l over corr higer Z
        
                # Get channel magnitudes separately for magnetometers and gradiometers
                inds_mag = [ind for ind,item in enumerate(raw_copy.info['chs']) if item['ch_name'].endswith('1')] # magnetometer indices
                inds_grad = [ind for ind,item in enumerate(raw_copy.info['chs']) if item['ch_name'].endswith(('2','3'))] # gradiometer indices
                
                amp_mag = np.max(data[inds_mag,:], axis=1) - np.min(data[inds_mag,:], axis=1)
                amp_mag_Z = zscore(amp_mag)  
                amp_mag_logZ = np.abs(zscore(np.log(amp_mag))) # the logarithmic scale is mainly to find dead channels
                amp_grad = np.max(data[inds_grad,:], axis=1) - np.min(data[inds_grad,:], axis=1) 
                amp_grad_Z = zscore(amp_grad)
                amp_grad_logZ = np.abs(zscore(np.log(amp_grad)))
                pow_mag = np.sqrt(np.sum(data[inds_mag,:] ** 2, axis=1))
                pow_mag_Z = zscore(pow_mag)
                pow_mag_logZ = np.abs(zscore(np.log(pow_mag)))
                pow_grad = np.sqrt(np.sum(data[inds_grad,:] ** 2, axis=1))
                pow_grad_Z = zscore(pow_grad)
                pow_grad_logZ = np.abs(zscore(np.log(pow_grad)))                               
                
                amp_all_Z = np.zeros(len(raw_copy.info['chs']))
                amp_all_Z[inds_mag] = amp_mag_Z
                amp_all_Z[inds_grad] = amp_grad_Z
                amp_all_logZ = np.zeros(len(raw_copy.info['chs']))
                amp_all_logZ[inds_mag] = amp_mag_logZ
                amp_all_logZ[inds_grad] = amp_grad_logZ
                pow_all_Z = np.zeros(len(raw_copy.info['chs']))
                pow_all_Z[inds_mag] = pow_mag_Z
                pow_all_Z[inds_grad] = pow_grad_Z
                pow_all_logZ = np.zeros(len(raw_copy.info['chs']))
                pow_all_logZ[inds_mag] = pow_mag_logZ
                pow_all_logZ[inds_grad] = pow_grad_logZ
        
                if method == 'corr':  # Based on local uncorrelation
                    feat_vec = chn_nei_uncorr_z
                    max_th = feat_vec > zscore_th
                elif method == 'amplitude': # Based on amplitude range
                    feat_vec = amp_all_Z
                    max_th = feat_vec > zscore_th
                elif method == 'power':  # Based on magnitude
                    feat_vec = pow_all_Z
                    max_th = feat_vec > zscore_th
                elif method == 'corr_power':  # Combine uncorrelation with magnitude
                    feat_vec = (chn_nei_uncorr_z + pow_all_Z) / 2
                    max_th = feat_vec > zscore_th
                elif method == 'amplitude_log': # Amplitude + log amplitude
                    feat_vec = np.vstack((amp_all_Z, amp_all_logZ))
                    max_th = feat_vec > zscore_th
                    max_th = np.logical_or(max_th[0,:], max_th[1,:])
                elif method == 'power_log':  # Power + log power
                    feat_vec = np.vstack((pow_all_Z, pow_all_logZ))
                    max_th = feat_vec > zscore_th
                    max_th = np.logical_or(max_th[0,:], max_th[1,:])
                
        
                bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
                if bad_chns:
                    print('Bad channels: %s' % bad_chns)
                else:
                    print('No bad channels:')
                # mark bad channels into raw file and plot data
                raw.info['bads'] = bad_chns                
                
                if plot:
                    print('Plotting data')                        
                    if method == 'power_log':
                        plt.figure(), plt.plot(feat_vec[0,:], label='power'), 
                        plt.plot(feat_vec[1,:], label='abs log power'), 
                        plt.axhline(zscore_th, color='k'), plt.legend(loc="upper left")  
                    elif method == 'amplitude_log':
                        plt.figure(), plt.plot(feat_vec[0,:], label='amplitude'), 
                        plt.plot(feat_vec[1,:], label='abs log amplitude'), 
                        plt.axhline(zscore_th, color='k'), plt.legend(loc="upper left")
                    else:
                        plt.figure(), plt.plot(feat_vec), plt.axhline(zscore_th)

                    raw.plot(n_channels=100, bad_color='r', highpass=f1, lowpass=f2, start=t1)
                  
                # write bad channels into text file
                print('Writing bad channels: %s to %s' % (bad_chns, out_fname))
                with open(out_fname, 'w') as f: 
                    for line in bad_chns: 
                        f.write(str(line) + '\n')
                f.close()
                
            self.raw.append(raw)
            print('\n')
                    
                

    def run_maxwell_filter(self, tSSS=True, erm=False, plot=False, overwrite=False):
        """
        Run Maxwell filtering (i.e., movement compensation and artifact reduction)
        
        Parameters
        ----------
        tSSS : bool
            Use temporal Signal Space Separation (default), otherwise SSS
        erm : bool
            Use empty room data (default False) 
        plot : bool
            Plot channels (default False)
        overwrite : bool
            Overwrite if already run (default False)
        """
        # fine calibration file (encodes site-specific information about sensor orientation and calibration)
        fine_cal_file = '/autofs/space/megraid_research/orsay/8/megdev/megsw-neuromag/databases/sss/sss_cal.dat'
        # crosstalk compensation file (reduces interference between Elekta’s co-located magnetometer and paired gradiometer sensor units)
        crosstalk_file = '/autofs/space/megraid_research/orsay/8/megdev/megsw-neuromag/databases/ctc/ct_sparse.fif'
 
        if erm:
            raw = mne.io.read_raw_fif(os.path.join(self.path_erm, self.erm_name[0]), preload=False)
            destination = None   
            coord_frame = 'meg'
            head_pos = None
        else:
            # use the first run as a reference for aligning runs to a common head position
            info = mne.io.read_info(os.path.join(self.path_speech, self.raw_names[0]), verbose='warning')
            destination = info['dev_head_t'] 
            coord_frame = 'head'
        
        os.makedirs(self.path_local, exist_ok=True)
        
        for raw_name in self.raw_names:
            if tSSS:
                if erm:
                    out_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0].replace('speech','erm') + '_tsss.fif')
                else:
                    out_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0] + '_tsss.fif')
            else:
                if erm:
                    out_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0].replace('speech','erm') + '_sss.fif')
                else:
                    out_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0] + '_sss.fif')
            
            if os.path.exists(out_fname) and not overwrite:
                print('Maxwell filter already run. No need to rerun.')
                if not erm:
                    info = mne.io.read_info(os.path.join(self.path_speech, raw_name), verbose='warning')
                    self.head_center = {} # initialize dict for saving head center data
                    self.head_center['radius'], self.head_center['origin_head'], \
                        self.head_center['origin_device']  = \
                        mne.bem.fit_sphere_to_headshape(info, units='m')
            else:
                if erm:
                    print('Running Maxwell filter for subject ' + self.sub + ', file ' + self.erm_name[0] + 
                          ' using the same preprocessing steps as for ' + raw_name)
                    print('Empty room data, no movement compensation')
                else:
                    print('Running Maxwell filter for subject ' + self.sub + ', file ' + raw_name)
                    print('Do movement compensation')
                    raw = mne.io.read_raw_fif(os.path.join(self.path_speech, raw_name), preload=False)
                    head_pos_file = os.path.join(self.path_speech, os.path.splitext(raw_name)[0]) + '_hp.pos'
                    head_pos = mne.chpi.read_head_pos(head_pos_file)
                    
                # read bad channels from text file
                bad_chns_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0] + '_bads.txt')
                bad_chns_f = open(bad_chns_fname, "r")
                bad_chns = bad_chns_f.read().splitlines()
                bad_chns_f.close()
                # check if the channel names match (i.e. whitespace between); fix if not
                if bad_chns and not bad_chns[0] in raw.info['ch_names']:
                    if ' ' in bad_chns[0]:
                        bad_chns = [ch_name.replace(' ', '') for ch_name in bad_chns]
                    else:
                        bad_chns = [ch_name.replace('MEG', 'MEG ') for ch_name in bad_chns]
                raw.info['bads'] = bad_chns
                
                if tSSS:
                    print('Using tSSS')
                    st_duration = 10; # default in MaxFilter™
                else:
                    print('Using SSS')
                    st_duration = None 
                                                                                                                            
                raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file, calibration=fine_cal_file, 
                                                           coord_frame=coord_frame, regularize='in', head_pos=head_pos, 
                                                           destination=destination, st_duration=st_duration, bad_condition='info')
                
                if plot: 
                    mne.viz.plot_head_positions(head_pos, mode='traces')
                    mne.viz.plot_alignment(raw.info, surfaces=[], coord_frame='meg', dig=True)
                    raw_sss.plot(highpass=1, lowpass=40)
                
                if not erm:
                    self.head_center = {} # initialize dict for saving head center data
                    self.head_center['radius'], self.head_center['origin_head'], \
                        self.head_center['origin_device']  = \
                        mne.bem.fit_sphere_to_headshape(raw.info, units='m')                            
                       
                raw_sss.save(out_fname, overwrite=True)
                    
            print('\n')


            
    def run_ICA(self, data_to_use='tsss', zscore_th=3.0, plot=True, overwrite=False):
        """
        Run Independent Component Analysis to reduce physiological artifacts (e.g., eye blinks and heart beats)
        
        Parameters
        ----------
        data_to_use : str
            Which data to se, options are 'raw', 'SSS', and 'tSSS' (default)
        zscore_th : float
            Another arbitrary threshold (default 3.0) 
        plot : bool
            Plot diagnostics (default True)
        overwrite : bool
            Overwrite if already run (default False)
        """
        print('Running ICA...')
                    
        if data_to_use == 'raw':                    
            fnames = self.raw_names           
        elif data_to_use == 'sss':
            fnames = [os.path.splitext(fname)[0] + '_sss.fif' for fname in self.raw_names]           
        elif data_to_use == 'tsss':
            fnames = [os.path.splitext(fname)[0] + '_tsss.fif' for fname in self.raw_names]
                            
        os.makedirs(self.path_local, exist_ok=True)
        self.ica = [] # initialize list
        for i,fname in enumerate(fnames):        
            out_fname = os.path.join(self.path_local, os.path.splitext(fname)[0] + '-ica.fif')
            if os.path.exists(out_fname) and not overwrite:
                print('ICA already run. No need to rerun.')
                ica = mne.preprocessing.read_ica(out_fname)
            else:
                raw = mne.io.read_raw_fif(os.path.join(self.path_local, fname), preload=False)
                raw_filt = raw.copy().load_data().filter(1,40)
                ica = mne.preprocessing.ICA(method='fastica', n_components=0.95, random_state=0, 
                                            max_iter=1000)
                if raw_filt.info['sfreq'] == 3000: decim=9 
                else: decim=3
                ica.fit(raw_filt, reject=dict(grad=4000e-13, mag=4e-12), decim=decim)
                eog_inds, eog_scores = ica.find_bads_eog(raw_filt, threshold=zscore_th)
                ecg_inds, ecg_scores = ica.find_bads_ecg(raw_filt, ch_name='MEG0141', 
                                                         method='correlation',
                                                         measure='correlation', 
                                                         threshold=0.4)

                print('EOG components: ' + str(eog_inds) + '\nECG components: ' + str(ecg_inds))
                ica.exclude = eog_inds + ecg_inds
                
                # plot ICs applied to raw data
                ica.plot_sources(raw_filt)
                
                # plot diagnostics: ICs applied to the averaged EOG and ECG epochs
                if plot:
                    eog_evoked = mne.preprocessing.create_eog_epochs(raw_filt).average()
                    ecg_evoked = mne.preprocessing.create_ecg_epochs(raw_filt).average()
                    ica.plot_sources(eog_evoked, title='Reconstructed latent sources, time-locked to EOG events')
                    ica.plot_sources(ecg_evoked, title='Reconstructed latent sources, time-locked to ECG events')
                
                    ica.plot_scores(eog_scores)
                    ica.plot_scores(ecg_scores)
                    ica.plot_properties(raw_filt, picks=eog_inds)
                    ica.plot_properties(raw_filt, picks=ecg_inds)
                
                raw_reconst = raw.copy().load_data()
                ica.apply(raw_reconst)
#                raw_reconst.plot(highpass=1, lowpass=40)
#                raw.plot(highpass=1, lowpass=40)
                ica.save(out_fname)
                
            self.ica.append(ica)
                


    def get_epochs(self, data_to_use='tsss', tmin=-0.2, tmax=2.0, fmin=None, fmax=40, notch=None,
                   reject=None, apply_ica=True, postfix=None, overwrite=False):
        """
        Extract epochs from raw data
        
        Parameters
        ----------
        data_to_use : str
            Which data to se, options are 'raw', 'SSS', and 'tSSS' (default)
        tmin : float
            Beginning of epoch with respect to events in seconds (default -0.2)
        tmax : float
            End of epoch with respect to events in seconds (default 2.0)
        fmin : float | None
            The lower pass-band edge frequency of the filter in Hz (default None)
        fmax : float | None
            The upper pass-band edge frequency of the filter in Hz (default 40)
        notch : float | None
            Frequency to notch filter in Hz (default None)
        reject : dict | None
            Reject epochs based on maximum peak-to-peak signal amplitude 
            (e.g., dict(grad=4000e-13, mag=4e-12); default None)
        apply_ica : bool
            Apply the ICA solution to the data (default True)
        postfix : str | None
            Save epochs using this postfix in the name
        overwrite : bool
            Overwrite if already run (default False)
        """
        print('Epoching...')
        
        if data_to_use == 'raw':
            raw_names = self.raw_names   
        elif data_to_use == 'sss':
            raw_names = [os.path.splitext(raw_name)[0] + '_sss.fif' for raw_name in self.raw_names]   
        elif data_to_use == 'tsss':
            raw_names = [os.path.splitext(raw_name)[0] + '_tsss.fif' for raw_name in self.raw_names]
                                 
        out_fname = '%s/%s-epo.fif' % (self.path_local, self.sub)
        if postfix:
            out_fname = '%s_%s-%s' % (out_fname.split('-')[0], postfix, out_fname.split('-')[1])

        self.epochs = [] # initialize list
        
        # if no overwrite requested, check if epoching already done 
        if not overwrite and os.path.exists(out_fname):
            print('Already epoched. No need to re-epoch.')
            self.epochs = mne.read_epochs(out_fname, proj=False)
            
        # otherwise do epoching
        else:            
            for raw_name in raw_names:
                                  
                raw = mne.io.read_raw_fif(os.path.join(self.path_local, raw_name), preload=False)
                # filter
                if notch:
                    picks = mne.pick_types(raw.info, meg=True)
                    raw_copy = raw.copy().load_data().filter(fmin,fmax).notch_filter(freqs=notch, picks=picks)
                else:
                    raw_copy = raw.copy().load_data().filter(fmin,fmax)
                
                # apply ICA to the raw data
                if apply_ica:
                    ica_fname = os.path.join(self.path_local, os.path.splitext(raw_name)[0] + '-ica.fif')
                    ica = mne.preprocessing.read_ica(ica_fname)
                    ica.apply(raw_copy)
                
                # get events
                fs = raw.info['sfreq'] # sampling frequency
                shortest_event = 1
                # min_duration = 2 / fs # shortest event (in samples) / sampling frequency
                min_duration = 0
                events = mne.find_events(raw_copy, stim_channel='STI101', verbose=True,
                                         min_duration=min_duration, initial_event=True,
                                         shortest_event=shortest_event)
                # if no events found, check the other stimulus channel
                if events.size == 0: 
                    events = mne.find_events(raw_copy, stim_channel='STI102', 
                                             min_duration=min_duration, 
                                             initial_event=True, 
                                             shortest_event=shortest_event)
                
                # set event IDs
                event_IDs = np.unique(events[:, 2])
                if len(event_IDs)==8:           
                    event_IDs_dict = {'Speech/A': event_IDs[0], 'Speech/B': event_IDs[1],
                                      'Jabber/A': event_IDs[2], 'Jabber/B': event_IDs[3],
                                      'MSS/B': event_IDs[4], 'SWS/B': event_IDs[5],
                                      'Noise/A': event_IDs[6], 'Noise/B': event_IDs[7]}
                elif len(event_IDs)==9: 
                    event_IDs_dict = {'Speech/A': event_IDs[1], 'Speech/B': event_IDs[2],
                                      'Jabber/A': event_IDs[3], 'Jabber/B': event_IDs[4],
                                      'MSS/B': event_IDs[5], 'SWS/B': event_IDs[6],
                                      'Noise/A': event_IDs[7], 'Noise/B': event_IDs[8]}
                
                # read epochs
                epochs = mne.Epochs(raw_copy, events=events, tmin=tmin, tmax=tmax, event_id=event_IDs_dict,
                                    reject=reject, baseline=(tmin, 0.0)).load_data()
                                                        
                self.epochs.append(epochs)
                
            # combine epochs into one object (if data in multiple files)
            self.epochs = mne.concatenate_epochs(self.epochs)
            # decimate from 3000Hz to 1000Hz if needed
            if fs != 1000: self.epochs.decimate(int(fs/1000))
#            if fs != 1000: self.epochs.resample(sfreq=1000, n_jobs=12)
            # save epochs
            self.epochs.save(out_fname, overwrite=overwrite)

                
                
    def forward_modeling(self, spacing='ico5', overwrite=False):
          
        print('Forward modeling subject ' + self.sub)
        print('Freesurfer ID: ' + self.fs_id)
        
        fname_src = os.path.join(self.path_fs, 'bem', '%s-%s-src.fif' % (self.fs_id_old, spacing))
        fname_src_alt = os.path.join(self.path_fs, 'bem', '%s-%s-%s-src.fif' % (self.fs_id_old, spacing[0:3], spacing[3]))
        fname_bem = os.path.join(self.path_fs, 'bem', '%s-%s-bem-sol.fif' % (self.fs_id_old, spacing))
        fname_bem_alt = os.path.join(self.path_fs, 'bem', '%s-%s-%s-bem-sol.fif' % (self.fs_id_old, spacing[0:3], spacing[3]))
        fname_trans = os.path.join(self.path_local, '%s_speech_raw_tsss-trans.fif' % self.sub)
        fname_trans_alt = os.path.join(self.path_local, '%s_speech-trans.fif' % self.sub)
        fname_fwd = os.path.join(self.path_local, '%s_speech_%s-fwd.fif' % (self.sub, spacing))
        
        if not os.path.exists(fname_src) and os.path.exists(fname_src_alt): fname_src = fname_src_alt
        if not os.path.exists(fname_src) or overwrite:
            src = mne.setup_source_space(subject=self.fs_id_new, subjects_dir=paths['fs'], spacing=spacing, n_jobs=12)
            src.save(fname_src, overwrite=overwrite)
                    
        if not os.path.exists(fname_bem) and os.path.exists(fname_bem_alt): fname_bem = fname_bem_alt
        if not os.path.exists(fname_bem) or overwrite:
            if not os.path.exists('%s/bem' % self.path_fs):
                subprocess.call('mkdir %s/bem' % self.path_fs, shell=True)
            mne.bem.make_watershed_bem(subject=self.fs_id_new, subjects_dir=paths['fs'], atlas=True, overwrite=False)
            # If "Command not found: mri_watershed", run the output command in tcsh after running
            # setenv SUBJECTS_DIR /autofs/cluster/transcend/MRI/WMA/recons/
            # source /usr/local/freesurfer/fs-stable6-env-autoselect
            
            os.system('rsync -avztcp %s/bem/watershed/%s_inner_skull_surface ' \
                      '%s/bem/inner_skull.surf' % (self.path_fs, self.fs_id_old, self.path_fs))
            os.system('rsync -avztcp %s/bem/watershed/%s_outer_skull_surface ' \
                      '%s/bem/outer_skull.surf' % (self.path_fs, self.fs_id_old, self.path_fs))
            os.system('rsync -avztcp %s/bem/watershed/%s_outer_skin_surface ' \
                      '%s/bem/outer_skin.surf' % (self.path_fs, self.fs_id_old, self.path_fs))
            
            model = mne.make_bem_model(subject=self.fs_id_new, ico=4, subjects_dir=paths['fs'], conductivity=[0.3])
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(fname_bem, bem)

        if not os.path.exists(fname_trans) and os.path.exists(fname_trans_alt): fname_trans = fname_trans_alt
        if os.path.exists(fname_fwd) and not overwrite:
            print('Forward model already exists. No need to rerun.')
            self.fwd = mne.read_forward_solution(fname_fwd)
        else:
            info = mne.io.read_info(os.path.join(self.path_local, os.path.splitext(self.raw_names[0])[0] + '_tsss.fif'), verbose='warning')
            self.fwd = mne.make_forward_solution(info, fname_trans, fname_src, fname_bem)
            mne.write_forward_solution(fname_fwd, self.fwd, overwrite=overwrite)
            
        print('\n')
            

            
    def compute_noise_cov(self, erm=True, method='empirical', fmin=None, fmax=None, 
                          notch=None, postfix='', rank=None, n_jobs=1, overwrite=False):
                
        if not postfix:
            postfix = method
        if erm:
            fname_cov = os.path.join(self.path_local, '%s_erm_%s-cov.fif' % (self.sub, postfix))
        else:
            fname_cov = os.path.join(self.path_local, '%s_speech_%s-cov.fif' % (self.sub, postfix))
            
        if not overwrite and os.path.exists(fname_cov):
            print('Noise covariance file already computed. No need to rerun.')
            self.ncov = mne.read_cov(fname_cov)
        else:
            if erm: # use empty room data
                raw_erm = [mne.io.read_raw_fif(os.path.join(self.path_local, 
                             os.path.splitext(raw_name)[0].replace('speech','erm') + 
                             '_tsss.fif')) for raw_name in self.raw_names]    
                raw_erm = mne.concatenate_raws(raw_erm)
                
                if fmin or fmax:
                    if notch:
                        picks = mne.pick_types(raw_erm.info, meg=True)
                        raw_erm = raw_erm.copy().load_data().filter(fmin,fmax).notch_filter(freqs=notch, picks=picks)
                    else:
                        raw_erm = raw_erm.copy().load_data().filter(fmin,fmax)
                
                # check that channel names match (i.e. white space or not)
                if not raw_erm.info['ch_names'][0] in self.epochs.info['ch_names']:
                    mapping=dict(zip(raw_erm.info['ch_names'][0:306], self.epochs.info['ch_names'][0:306]))
                    raw_erm.rename_channels(mapping)
                                            
                ncov = mne.compute_raw_covariance(raw_erm, method=method, rank=rank, n_jobs=n_jobs) 
            else: # use prestimulus baselines instead
                ncov = mne.compute_covariance(self.epochs, method=method, rank=rank, tmax=0, n_jobs=n_jobs)
                           
            mne.write_cov(fname_cov, ncov)
            self.ncov = ncov
        
            

    def make_inv_operator(self, postfix='', overwrite=False):

        if not postfix:
            postfix = 'speech'
        fname_inv = os.path.join(self.path_local, '%s_%s-inv.fif' % (self.sub, postfix))
        if not overwrite and os.path.exists(fname_inv):
            print('Inverse operator already exists. No need to rerun.')
            self.inv = mne.minimum_norm.read_inverse_operator(fname_inv)
        else:
            self.inv = mne.minimum_norm.make_inverse_operator(self.epochs.info, 
                                                              self.fwd, self.ncov, 
                                                              loose=0.2, depth=0.8,
                                                              rank='info')
            mne.minimum_norm.write_inverse_operator(fname_inv, self.inv)


                
    ########################################################################### 
    ### helpers
    ###########################################################################
    
    def write_bad_channels(self, erm=False):
                      
        for i,raw in enumerate(self.raw):                        
            os.makedirs(self.path_local, exist_ok=True)
                       
            out_fname = os.path.join(self.path_local, os.path.splitext(self.raw_names[i])[0] + '_bads.txt')  
            bad_chns = self.raw[i].info['bads']
                
            print('Writing bad channels: ' + str(bad_chns) + ' to ' + out_fname) 
            
            # write bad channels into text file
            with open(out_fname, 'w') as f: 
                for line in bad_chns: 
                    f.write(str(line) + '\n')
            f.close()
            
                        
    def write_ica(self, data_to_use='tsss'):
        
        if data_to_use == 'raw':
            raw_names = self.raw_names         
        elif data_to_use == 'sss':
            raw_names = [os.path.splitext(raw_name)[0] + '_sss.fif' for raw_name in self.raw_names]        
        elif data_to_use == 'tsss':
            raw_names = [os.path.splitext(raw_name)[0] + '_tsss.fif' for raw_name in self.raw_names]
        
        for i,ica in enumerate(self.ica):              
            out_fname = os.path.join(self.path_local, os.path.splitext(raw_names[i])[0] + '-ica.fif')   
            ica.exclude = list(set(ica.exclude))
            print('Writing ICA solution to ' + out_fname)
            print('Excluding components: ' + str(ica.exclude))
            ica.save(out_fname)
            
            
    def get_head_movement(self, plot=False):
        from nibabel.eulerangles import mat2euler
        # from mne.fixes import einsum
        self.head_movement = []
        for raw_name in self.raw_names:
            head_pos_file = os.path.join(self.path_speech, os.path.splitext(raw_name)[0]) + '_hp.pos'
            head_pos = mne.chpi.read_head_pos(head_pos_file)
            if plot: mne.viz.plot_head_positions(head_pos, mode='traces')
            trans, rot, t = mne.chpi.head_pos_to_trans_rot_t(head_pos)
            use_trans = np.einsum('ijk,ik->ij', rot[:, :3, :3].transpose([0, 2, 1]), -trans) * 1000
            use_rot = rot.transpose([0, 2, 1])
            rotations = np.zeros(use_trans.shape)    
            for i, mat2 in enumerate(use_rot):
                rads2 = mat2euler(mat2) # rotations in radians around z, y, x axes
                yaw, roll, pitch = np.degrees(rads2)
                rotations[i,:] = pitch, roll, yaw
            mot_all = np.concatenate([np.expand_dims(t,1), use_trans, rotations], axis=1) # This is for output to a file, we don't need time in the array
            mot_parms = np.diff(mot_all[:, 1:7], axis=0)   
            # Get motion at each measurement (enorm of the 6 motion parms)
            mot_norms = np.linalg.norm(mot_parms, axis=1)   
            motion_mean = np.mean(mot_norms)
            motion_std = np.std(mot_norms, ddof=1)
            self.head_movement.append((motion_mean, motion_std))
            
            # norm = np.linalg.norm(use_trans,axis=1)
            # self.head_movement.append(norm.std())
            
                   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
