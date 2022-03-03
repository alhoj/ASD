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


def r_from_p(p, n):
    """ 
    Returns pearson's r corresponding to p-value (two-tailed) and sample size n.
    
    Parameters
    ----------
    p : float
        P-value
    n : int
        Sample size
        
    Returns
    -------
    r : float
        Pearson's correlation coefficient
    """
      
    rhos = np.arange(0,1,0.001)
    dist = scipy.stats.beta(n/2-1, n/2-1, loc=-1, scale=2)
    pvals = np.zeros(len(rhos))
    for i,rho in enumerate(rhos):
        pvals[i] = 2*dist.cdf(-abs(rho))
    r = rhos[pvals==np.max(pvals[pvals < p])][0]
    return r


def p_from_r(r, n):
    """ 
    Returns p-value (two-tailed) corresponding to pearson's r with sample size n.
    
    Parameters
    ----------
    r : float
        Pearson's correlation coefficient
    n : int
        Sample size
        
    Returns
    -------
    p : float
        P-value
        
    """
    
    t = r*np.sqrt((n-2)/(1-r**2))
    p = scipy.stats.t.sf(np.abs(t), n-2)*2
    return p


def t_from_p(p, n, tail='two'):
    """ 
    Returns t-statistic corresponding to p-value and sample size n.
    
    Parameters
    ----------
    p : float
        P-value
    n : int
        Sample size
    tail : 'one' | 'two'
        One- of two-tailed test (default 'two')
        
    Returns
    -------
    t : float
        T-statistic
        
    """
    
    if tail=='two':
        p /= 2
    t = abs(scipy.stats.distributions.t.ppf(p, n-1))
    return t



def get_signif_level(data, p=0.05, tail='greater'):
    """     
    Estimates one-sample significance level corresponding to p-value
    using wilcoxon signed rank test.
    
    Parameters
    ----------
    data : array, shape (n_samples)
        Group data
    p : float
        P-value (default 0.05)
    tail : 'greater' | 'less' | 'two-sided'
        The alternative hypothesis to be tested (default 'greater')
        
    Returns
    -------
    signif : float
        Significance level
        
    """
    checkVals = np.arange(0,np.max(data),np.max(data)/1000)
    pvals = np.zeros(len(checkVals))
    for i,checkVal in enumerate(checkVals):
        _,pvals[i] = scipy.stats.wilcoxon(data-checkVal, alternative=tail, mode='approx')
    signif = checkVals[pvals==np.max(pvals[pvals < p])][0]
    return signif



def pooled_stdev(s1, s2):
    """ 
    Calculate pooled standard deviation for sampless s1 and s2.
    
    Parameters
    ----------
    s1 : array, shape (n_samples)
        First sample
    s2 : array, shape (n_samples)
        Second sample
        
    Returns
    -------
    stdev : float
        Standard deviation
        
    """
    
    stdev = np.sqrt(((len(s1)-1) * np.var(s1) + (len(s2)-1) * 
                     np.var(s2)) / (len(s1)+len(s2)-2))
    return stdev



def compare_corr_coefs(r1, r2, n1, n2):
    """ 
    Compares correlation coefficients using Fisher's z-transformation.
    Returns z-score and p-value for the difference between the correlations.
    
    Parameters
    ----------
    r1 : float
        First correlation coefficient
    r2 : float
        Second correlation coefficient
    n1 : int
        First sample size
    n2 : int | None
        Second sample size. If n2 is None, assumes both r1 and r2 are 
        from the first sample.
        
    Returns
    -------
    z : float
        Z-score
    p : float
        P-value
        
    """
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



def DAR_MI(sig_phase, sig_phase_imag, sig_amp, amp_fq_range, fs):
    """ 
    Calculates phase-amplitude coupling (PAC) modulation index based on
    driven auto-regressive models (as in Dupre la Tour et al. 2017).
    
    Parameters
    ----------
    sig_phase : array, shape (n_epochs, n_points)
        Signal for phase (aka driver) 
    sig_phase_imag : array, shape (n_epochs, n_points)
        Imaginary part of the signal for phase
    sig_amp : array, shape (n_epochs, n_points)
        Signal for amplitude
    amp_fq_range : array
        Frequency range (for amplitude)
    fs : int
        Sampling frequency
        
    Returns
    -------
    MI : array, shape (n_frequencies, n_epochs)
        Modulation indices
        
    """    
    
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




def canolty_MI(sig_phase, sig_amp):
    """ 
    Calculates phase-amplitude coupling modulation index 
    as in Canolty et al. 2006.
    
    Parameters
    ----------
    sig_phase : array, shape (n_epochs, n_points)
        Signal for phase (aka driver) 
    sig_amp : array, shape (n_epochs, n_points)
        Signal for amplitude
        
    Returns
    -------
    MI : array, shape (n_frequencies_phase, n_frequencies_amplitude, n_epochs)
        Modulation indices
        
    """
    
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




def ozkurt_MI(sig_phase, sig_amp):
    """ 
    Calculates phase-amplitude coupling modulation index 
    as in Ozkurtr et al. 2011.
    
    Parameters
    ----------
    sig_phase : array, shape (n_epochs, n_points)
        Signal for phase (aka driver) 
    sig_amp : array, shape (n_epochs, n_points)
        Signal for amplitude
        
    Returns
    -------
    MI : array, shape (n_frequencies_phase, n_frequencies_amplitude, n_epochs)
        Modulation indices
        
    """
    
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


# convert (configuration) module into a dict
def module_to_dict(module):
    """ 
    Converts module into a dictionary with all the module attributes
    converted into dictionary items.
    
    Parameters
    ----------
    module : module
        Module to be converted
        
    Returns
    -------
    dictionary : dict
        Dictionary with module attributes converted into items.
        
    """
    dictionary = {}
    for attr in dir(module):
        if not attr.startswith('_'):
            dictionary[attr] = getattr(module, attr)
    return dictionary