#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:37:01 2022

@author: ja151
"""

import sys
sys.path.append('/autofs/cluster/transcend/jussi/scripts')
import numpy as np
import mne
from MNEprepro import MNEprepro


# First, I would recommend changing the 'local' path in the beginning of 
# the MNEprepro to your own working directory so that the output go there.

# Define subject ID
sub_ID = '063101'

# Input subject ID to MNEprero
sub = MNEprepro(sub_ID)

## Step 1: Detect bad channels
# Detecs bad channels automatically but also enables adding/removing bad channels
# interactively based on visual inspection. To add/remove, click the channel/signal 
# on the raw data plot (bad channels marked in red). Also plots the diagnostics 
# based on which the automatic detection was done. Marks the bad channels to 
# the raw fif files and also saves them to text files. If overwrite=False and 
# bad channel detection run alread, reads the bad channels from the text file(s).
# See https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html
sub.detect_bad_channels()

## Step 2: Maxwell filtering (movement compensation and artifact reduction)
# By default, temporal Signal Space Separation (tSSS) is done.
# Saves the output to the local directory with _tsss.fif extension.
# Should also be run to the empty room data (with identical parameters) 
# if the noise covariance matrix used in the inverse operator is going to be 
# estimated from the empty room data.
# See https://mne.tools/dev/auto_examples/preprocessing/movement_compensation.html
sub.run_maxwell_filter()

## Step 3: Independent Component Analysis for reducing physiological artifacts
# By default, uses tSSS data and plots diagnostics.
# Detects eye blinks/movement and heart beat components automatically but 
# visual inspection highly recommended. To add/remove, click the component 
# on the plot. Saves the output to the local directory with -ica.fif extension.
# See https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
sub.run_ICA()

## Step 4: Extract epochs from raw data
# By default, uses tSSS data and applies the ICA solution.
# Extract epochs from "tmin" to "tmax" with respect to all events found in raw data.
# Uses FIR bandpass filter with lower and upper pass-band egde defined by "fmin" and "fmax", respectively.
# Notch filter at 60 Hz also recommended to filter out line frequency (if fmax > 60 Hz).
# By default, does not reject epochs based on maximum peak-to-peak signal amplitude.
# Saves the output to the local directory with an optional "postfix" (i.e., subID_"postfix"-epo.fif).
sub.get_epochs()

## Step 5: Forward modeling
# To compute the forward operator you need 1) -trans.fif file that contains the MEG-MRI coregistration info,
# 2) source space (-src.fif), and 3) BEM surfaces (for EEG, inner skull, outer skull, and skin surfaces are needed; 
# for MEG, inner skull surface is enough).
# If source space and/or BEM are not precomputed, the script will compute them.
# See https://mne.tools/0.23/auto_tutorials/forward/30_forward.html
sub.forward_modeling()

## Step 6: Noise covariance matrix
# Computes noise covariance matrix required for the minimum-norm inverse solution.
# By default, uses empty room data and empirical method where the sample covariance will be computed.
# Filtering (bandpass) should be identical to that of the epochs.
# Also, if Maxwell filtering and/or ICA was applied on the subject data, it should be applied also 
# to empty room data before estimating the noise covariance (meaning that if the subject data was 
# collected in e.g. three runs with different ICA components excluded in each run, the empty room data 
# should be processed three times resulting in three matching processed empty room data files).
# Assumes that the processed empty room data file(s) are in the local directory and have 'erm' in the name
# See https://mne.tools/stable/generated/mne.compute_covariance.html
sub.compute_noise_cov()

## Step 7: Assemble inverse operator
# Gets the forward solution and noise covariance matrix from the self-instance.
# By default, uses loose orientation constraint of 0.2 and depth weighting of 0.8.
# Saves the output to the local directory with an optional "postfix" (i.e., subID_"postfix"-inv.fif).
# To compute the forward operator you need 1) -trans.fif file that contains the MEG-MRI coregistration info,
# 2) source space (-src.fif), and 3) BEM surfaces (for EEG, inn
# See https://mne.tools/stable/generated/mne.minimum_norm.make_inverse_operator.html
sub.make_inv_operator()









