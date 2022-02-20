#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:07:19 2021

@author: ja151
"""

import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt

paths = {'in': '/autofs/cluster/transcend/MEG/speech/',
         'out': '/local_mount/space/hypatia/1/users/jussi/speech/',
         'cluster': '/autofs/cluster/transcend/jussi/',
         'erm': '/autofs/cluster/transcend/MEG/erm/',
         'fs': '/autofs/cluster/transcend/MRI/WMA/recons/'
         }

fns = ['SWS', 'MSS']

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
for fn,ax in zip(fns, axs):
    file_path = '%s/stimuli/%s.wav' % (paths['cluster'], fn)
    y,sr = librosa.load(file_path, sr=None, mono=True, offset=0, duration=None)
    
    hop_length = 1024
    stft = librosa.stft(y, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='linear', 
                                   hop_length=hop_length, ax=ax)
    ax.set_title(fn, fontweight='bold')
    ax.set_ylim(0, 4000)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.label_outer()
plt.tight_layout()
fig.savefig('%s/figures/spectrogram_%s.png' % (paths['cluster'], 
                                               ('_').join(fns)), dpi=500)
plt.close()