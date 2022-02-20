#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 03:37:06 2020

@author: ja151
"""

import numpy as np
import scipy
import sys
if '/autofs/cluster/transcend/jussi/tools/' not in sys.path:
    sys.path.append('/autofs/cluster/transcend/jussi/tools/')
from pactools.dar_model.dar import DAR
from pactools.utils.maths import next_power2
import math

# estimate pearson's correlation coefficient corresponding to 
# given p-value and sample size 
def r_from_p(p,n):
    rhos = np.arange(0,1,0.001)
    dist = scipy.stats.beta(n/2-1, n/2-1, loc=-1, scale=2)
    pvals = np.zeros(len(rhos))
    for i,rho in enumerate(rhos):
        pvals[i] = 2*dist.cdf(-abs(rho))
    r = rhos[pvals==np.max(pvals[pvals < p])][0]
    return r

# calculate p-value corresponding to pearson's correlation coefficient
# and sample size 
def p_from_r(r,n):
    t = r*np.sqrt((n-2)/(1-r**2))
    p = scipy.stats.t.sf(np.abs(t), n-2)*2
    return p

# estimate significance level (group mean) 
# corresponding to given p-value  
# using wilcoxon signed rank test 
def get_signif_level(data, p=0.05, tail='greater'):
    checkVals = np.arange(0,np.max(data),np.max(data)/1000)
    pvals = np.zeros(len(checkVals))
    for i,checkVal in enumerate(checkVals):
        _,pvals[i] = scipy.stats.wilcoxon(data-checkVal, alternative=tail, mode='approx')
    signif = checkVals[pvals==np.max(pvals[pvals < p])][0]
    return signif


# calculate pooled standard deviation
# a, b = samples
def pooled_stdev(a, b):
    stdev = np.sqrt(((len(a)-1) * np.var(a) + (len(b)-1) * np.var(b)) / (len(a)+len(b)-2))
    return stdev

# correlation coefficients using Fisher's z-transformation
# returns z-score and p-value for the difference between the correlations
# if n2=None, do within-group test
def compare_corr_coefs(r1, r2, n1, n2):
    # Fisher's z-transformation
    z1 = math.atanh(r1)
    z2 = math.atanh(r2)
    
    # expected standard deviation
    if not n2:
        sd = np.sqrt(1/(n1-3))
    else:
        sd = np.sqrt(1/(n1-3) + 1/(n2-3))
    
    # z-score and p-value for the difference between the correlations
    z = (z1 - z2)/sd
    p = (1 - scipy.stats.norm.cdf(abs(z))) * 2
    return z, p

# calculate phase-amplitude coupling (PAC) modulation index based on 
# driven auto-regressive models (as in Dupre la Tour et al. 2017)
def DAR_MI(sig_phase, sig_phase_imag, sig_amp, amp_fq_range, fs):    
    model = DAR(ordar=10, ordriv=1)
    n_epochs = sig_phase.shape[0]
    MI = np.zeros((amp_fq_range.size, n_epochs))
    for i in range(n_epochs):
        model.fit(fs=fs, sigin=sig_amp[i], 
                  sigdriv=sig_phase[i], 
                  sigdriv_imag=sig_phase_imag[i])
        # estimate the length of the padding for the FFT
        if len(amp_fq_range) > 1:
            delta_f = np.diff(amp_fq_range).mean()
            n_fft = next_power2(fs / delta_f)
        else:
            n_fft = 1024                                
        # get PSD difference
        spec, _, _, _ = model._amplitude_frequency(n_fft=n_fft)                            
        # KL divergence for each phase, as in [Tort & al 2010]
        n_freq, n_phases = spec.shape
        spec = 10. ** (spec / 20.)
        spec = spec / np.sum(spec, axis=1)[:, None]
        spec_diff = np.sum(spec * np.log(spec * n_phases), axis=1)
        spec_diff /= np.log(n_phases)                            
        # crop the spectrum to amp_fq_range
        freqs = np.linspace(0, fs // 2, spec_diff.size)
        spec_diff = np.interp(amp_fq_range, freqs, spec_diff)
        MI[:,i] = spec_diff
    return MI


# calculate PAC modulation index as in Canolty et al. 2006
def canolty_MI(sig_phase, sig_amp):
    n_phase_fq = sig_phase.shape[0]
    n_amp_fq = sig_amp.shape[0]
    n_epochs = sig_phase.shape[1]
    MI = np.zeros((n_phase_fq, n_amp_fq, n_epochs))
    for i_phase in range(n_phase_fq):
        for i_amp in range(n_amp_fq):                
            # preprocess the phase array
            phase_preprocessed = np.exp(1j * sig_phase[i_phase])
            amplitude = sig_amp[i_amp]
            z_array = amplitude * phase_preprocessed
            MI[i_phase,i_amp] = np.abs(z_array.mean(1))
    return MI


# calculate PAC modulation index as in Ozkurtr et al. 2011
def ozkurt_MI(sig_phase, sig_amp):
    n_amp_fq = sig_amp.shape[0]
    n_epochs = sig_amp.shape[1]
    n_tps = sig_amp.shape[2]
    sig_amp_shape = sig_amp.shape
    sig_amp = sig_amp.reshape(n_amp_fq*n_epochs, n_tps)
    norms = np.zeros(n_amp_fq*n_epochs)
    for i in range(len(norms)):
        # Euclidean norm when x is a vector, Frobenius norm when x
        # is a matrix (2-d array). More precise than sqrt(squared_norm(x)).
        nrm2, = scipy.linalg.get_blas_funcs(['nrm2'], [sig_amp[i]])
        norms[i] = nrm2(sig_amp[i])
    norms = norms.reshape(n_amp_fq, n_epochs) 
    sig_amp = sig_amp.reshape(sig_amp_shape)
    
    n_phase_fq = sig_phase.shape[0]
    MI = np.zeros((n_phase_fq, n_amp_fq, n_epochs))
    for i_phase in range(n_phase_fq):
        for i_amp in range(n_amp_fq):                
            # preprocess the phase array
            phase_preprocessed = np.exp(1j * sig_phase[i_phase])
            amplitude = sig_amp[i_amp]            
            norm = norms[i_amp]
            z_array = amplitude * phase_preprocessed
            MI[i_phase,i_amp] = np.abs(z_array.mean(1))
            MI[i_phase,i_amp] *= np.sqrt(amplitude.shape[1]) / norm
    return MI