import mne 
import sys
from glob import glob
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

def data_read(all_file_path):
    data=mne.io.read_raw_gdf(all_file_path,preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
    print(data,"\n")
    #picks = mne.pick_channels(data.info['ch_names'], ['MEG 2443', 'MEG 2442', 'MEG 2441'])
    #print("\n Channle name = ",picks)
    data=data.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    print("data information =\n",data.info)
    print("HIIIII",data.info['dig'])
   # print("data description =\n",data.describe())
    data=data.filter(1,20)
    print("Chanel Names = \n",data.ch_names,"\n")
    data.plot() 
   
    print("\n",data.annotations,"\n")
    events=mne.events_from_annotations(data)
    print("Events are=\n",events[0][0:20],"\n")
    data.set_eeg_reference()
    data.filter(l_freq=0.5,h_freq=45)
    
    event_dict={
        'reject=':1,
        'eye move=':2,
        'eye open=':3,
        'eye close=':4,
        'new run=':5,
        'new trial':6,
        'class 1(Cue onset left)=':7,
        'class 2(Cue onset right)=':8,
        'class 3(Cue onset foot)=':9,
        'class 4(Cue onset tongue)=':10
    }
    for keys, values in event_dict.items(): 
        print(keys ,values)
        
    fig = mne.viz.plot_events(events[0], event_id=event_dict, sfreq=data.info['sfreq'],first_samp=data.first_samp,on_missing='ignore')
    fig.subplots_adjust(right=0.7)
    epochs = mne.Epochs(data,events[0], event_id=[7,8 ,1,2],tmin= -0.1, tmax=0.7, preload=True,on_missing='ignore')
    
    
    #evoked=mne.Evoked(epochs, condition=None, proj=True, kind='average', allow_maxshield=False,  verbose=None)
    '''for keys, values in event_dict.items(): 
        print(keys ,values)
        evoked[keys] = epochs['values'].average()
        print(evoked_keys)'''
    evoked_0 = epochs['7'].average()
    evoked_1 = epochs['8'].average()
    evoked_2 = epochs['1'].average()
    evoked_3 = epochs['2'].average()
    #left,right,foot,tongue
    dicts={'class0':evoked_0,'class1':evoked_1,'reject':evoked_2}
    
    fig= mne.viz.plot_compare_evokeds(dicts)
    #fig.subplots_adjust(right=0.7)
    labels=epochs.events[:,-1]
    print("\nLabels are =\n",labels)
    features=epochs.get_data()
    print(epochs.get_data().shape)

    clf = make_pipeline( StandardScaler(),LogisticRegression(solver='liblinear'))  # liblinear is faster than lbfg)
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=0,verbose=True)

# Fit classifiers on the epochs where the stimulus was presented to the left.
# Note that the experimental condition y indicates auditory or visual
    print(time_gen.fit(X=epochs[7,8,1,2].get_data(), y=epochs[7,8,1,2].events[:, 2] ))
    
    scores = time_gen.score(X=epochs[7,8,1,2].get_data(),y=epochs[7,8,1,2].events[:, 2])
    print(scores)
    
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower',extent=epochs.times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Condition: "Right"\nTesting Time (s)',)
    ax.set_ylabel('Condition: "Left"\nTraining Time (s)')
    ax.set_title('Generalization across time and condition', fontweight='bold')
    fig.colorbar(im, ax=ax, label='Performance (ROC AUC)')
    plt.show()
    return features,labels
    
    pass


def main():
    all_file_path=glob('/home/soumen/Documents/EEG Data/BCICIV_2a_gdf/*')
    #all_file_path=glob('/home/soumen/Documents/EEG Data/BCICIV_2b_gdf/*')
    print("No of Files available= ",len(all_file_path),"\n")
    for i in range (0,len(all_file_path)):
        print(all_file_path[i])
    print("\n")
    eeg_gdf_data_epochs_array=[data_read(i) for i in all_file_path]
    


main()