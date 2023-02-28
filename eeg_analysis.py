import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys
import mne 
import glob
from scipy import stats



all_file_path=glob.glob('/home/soumen/Documents/EEG Data/sampleeegdata/*')
#print("\n",all_file_path,"\n")
print("No of Files available= ",len(all_file_path),"\n")
for i in range (0,len(all_file_path)):
    print(all_file_path[i])
print("\n")

def read_data(file_path):
    datax=mne.io.read_raw_brainvision(file_path,eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto', scale=1.0, preload=True, verbose=None)
    print("HIIII",datax)
    datax.set_eeg_reference()
    datax.filter(l_freq=1,h_freq=45)
    epochs=mne.make_fixed_length_epochs(datax,duration=25,overlap=0)
    epochs=epochs.get_data()
    datax.plot_psd(fmax=50)
    datax.plot(duration=5, n_channels=1)
    return epochs #trials,channel,length



def main():
    read_data(all_file_path[1])
    pass

main()