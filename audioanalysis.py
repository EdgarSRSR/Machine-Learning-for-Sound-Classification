# the objective of this file is to experiment with multiple parameters and characteristics
# for the analysis of audio files. It does different analysis like calculating spectral centroid,
# fast fourier transforms or Mffcs. It also plot different types of spectograms.
# The analysis is done thanks to libraries like librosa who are created for manipulation of audio
# The idea of this file is to get different parameters and commenting out whatever functions that
# arre not needed.

import os
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import scipy as sp
import librosa
import librosa.display
import glob
import csv 

inp = "nameofaudiofile.wav"
#audio is decoded as a time series where y is one dimensional numpY point array
# and sr is the sampling rate of y
y, sr = librosa.load(inp)

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

# Compute MFCC features from the raw signal, it calculates 13 mfcc in one file
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13).T,axis=0) 
# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])


#compute short-time fourier transform
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

#Compute Spectral centroid
cent = librosa.feature.spectral_centroid(y = y, sr = sr)

#Compute Spectral Roll-Off, approximate maximun frequencies with roll percent =.9
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)

#Compute Zero Crossing Rate
zerocross = librosa.feature.zero_crossing_rate(y)

#Compute Short Time Energy

feature = mfccs
label = "landmower"

print("mel-frequency cepstral coefficients: ")
print( [feature, label])
##print("beat features: ")
##print(beat_features)
print("Short Time Furier Transform: ")
print(D)
print("Spectral centroid: ")
print(cent)
print("spectral Roll-Off: ")
print(rolloff)
print("Zero crossing rate: ")
print(zerocross)
print("energy")

create a csv file with information
with open('results.csv', mode = 'w') as results_file:
	results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
	results_writer.writerow(['example','mfccs', 'zero Crossing Rate','label'])
	results_writer.writerow(['21',feature, zerocross,label])


#visualise waveplot
plt.figure(1)
librosa.display.waveplot(y,sr=sr)
plt.title('mono')
waveplotimg = inp + 'waveplot' + '.jpg'
plt.savefig(waveplotimg, bbox_inches=None, pad_inches=0)


#visualize short-time fourier transform power spectrum
plt.figure(2, figsize=(2.24,2.24))

#plt.subplot(4, 2, 1)
librosa.display.specshow(D, x_axis='time',y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
specplotimg = inp + 'specplot' + '.jpg'
plt.savefig(specplotimg, box_inches=None, pad_inches=0)


plt.figure(3, figsize=(10, 4))
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                     fmax=8000)
librosa.display.specshow(librosa.power_to_db(S,
                                              ref=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
melplotimg = inp + 'melplot' + '.jpg'
plt.savefig(melplotimg, box_inches=None, pad_inches=0)
plt.show()
