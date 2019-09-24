# This file jumps into a directory containing audio files in either mp3 or wav formats and
# goes through each file calculating the Mfcc and creating spectograms. The Mffc's information
# is then stored in a csv file. This information is useful for crating data sets for Machine Learning.

import os
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import scipy as sp
import librosa
import librosa.display
import glob
import csv

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512
label = "0"
files_path = '/Users/edgarsr/Desktop/ТГУ/thesis/sounds/drones'
#print(os.listdir(files_path))
#create mfccs, number of mfccs calculated is 13 for each file
#for filename in os.listdir(files_path):
    
#    if filename.endswith(".mp3") or filename.endswith(".wav"):
        #load data
#        y, sr = librosa.load(filename)
#        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13).T,axis=0) 
#        feature = mfccs
#        with open('results.csv', mode = 'a') as results_file:
#        	results_writer = csv.writer(results_file, delimiter=',', quotechar='"')
#        	results_writer.writerows([feature,label])


#create spectrograms
for filename in os.listdir(files_path):
    
    if filename.endswith(".mp3") or filename.endswith(".wav"):
    	inp = filename
    	y,sr = librosa.load(inp)
    	D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    	plt.figure(1, figsize=(2.24,2.24))
    	librosa.display.specshow(D)
    	#plt.colorbar(format='%+2.0f dB')
    	#plt.title('Linear-frequency power spectrogram')
    	specplotimg = inp + 'specplot' + '.jpg'
    	plt.savefig(specplotimg, box_inches=None, pad_inches=0)    
#y, sr = librosa.load("dronetest1.mp3")
#hop_length = 512
#mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13).T,axis=0) 
#zerocross = librosa.feature.zero_crossing_rate(y)
#feature = mfccs
#label = "1"

#with open('results.csv', mode = 'a') as results_file:
#	results_writer = csv.writer(results_file, delimiter=',', quotechar='"')
#	results_writer.writerow(['mfccs','label'])
#	results_writer.writerows([feature,label])

#with open('results.csv', mode = 'a') as results_file:
#	results_writer = csv.writer(results_file, delimiter=',', quotechar='"')
#	results_writer.writerow(['mfccs','label'])
#	results_writer.writerows([feature,label])
