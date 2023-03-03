from glob import glob
import os 
import mne
import matplotlib.pyplot as plt
import sys
import numpy as np 
import pandas as pd
from scipy import stats
from sklearn import metrics
import colorama
from colorama import Fore ,Back, Style
from simple_colors import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix



def data_read(all_file_path):
    data=mne.io.read_raw_edf(all_file_path,preload=True)
    data.set_eeg_reference()
    data.filter(l_freq=0.5,h_freq=45)
    epochs=mne.make_fixed_length_epochs(data,duration=5,overlap=1)
    array=epochs.get_data()
    return array
    pass


def epochs_label_annotation(normal_patients ,schizophrenia_patient):
    print("\n normal_patients_data_dimension=",normal_patients[0].shape)
    print("\n schizophrenia_patients_data_dimension=",schizophrenia_patient[0].shape)
    normal_patients_epoch_labels=[len(i)*[0] for i in normal_patients]
    schizophrenia_patient_epoch_labels=[len(i)*[1] for i in schizophrenia_patient]
    #print(len(normal_patients_epoch_labels),len(schizophrenia_patient_epoch_labels))

    # Combine both the category for "data" & "Labels"
    combined_data_list=normal_patients+schizophrenia_patient
    combined_label_list=normal_patients_epoch_labels+schizophrenia_patient_epoch_labels

    # subjects are clustered into different groups instead of random sampling
    group_list=[[i]*len(j) for i,j in enumerate(combined_data_list)]
    print("length of group list =",len(group_list))

    # convert list to array
    data_array=np.vstack(combined_data_list)
    
    label_array=np.hstack(combined_label_list)
    group_array=np.hstack(group_list)
    print(data_array.shape,label_array.shape,group_array.shape)
    Feature_extraction(data_array,label_array,group_array)
    pass


def Feature_extraction(data,label_array, group_array):
    mean=np.mean(data,axis=-1)
    print(Fore.GREEN+"mean =\n",mean)
    print(Fore.YELLOW+"Shape =",mean.shape,"\n")

    std=np.std(data,axis=-1)
    print(Fore.GREEN+"Standard Deviation =\n",std)
    print(Fore.YELLOW+"Shape =",std.shape,"\n")

    ptp=np.ptp(data,axis=-1)
    print(Fore.GREEN+"peak to peak =\n",ptp)
    print(Fore.YELLOW+"Shape =",ptp.shape,"\n")

    var=np.var(data,axis=-1)
    print(Fore.GREEN+"Variance =\n",var)
    print(Fore.YELLOW+"Shape =",var.shape,"\n")

    minimum=np.min(data,axis=-1)
    print(Fore.GREEN+"minimum =\n",minimum)
    print(Fore.YELLOW+"Shape =",minimum.shape,"\n")

    maximum=np.max(data,axis=-1)
    print(Fore.GREEN+"maximum =\n",maximum)
    print(Fore.YELLOW+"Shape =",maximum.shape,"\n")

    arg_min=np.argmin(data,axis=-1)
    print(Fore.GREEN+"argmin=\n",arg_min)
    print(Fore.YELLOW+"Shape =",arg_min.shape,"\n")

    arg_max=np.argmax(data,axis=-1)
    print(Fore.GREEN+"argmin=\n",arg_max)
    print(Fore.YELLOW+"Shape =",arg_max.shape,"\n")

    rms_val=np.sqrt(np.mean(data**2,axis=-1))
    print(Fore.GREEN+"rms_val=\n",rms_val)
    print(Fore.YELLOW+"Shape =",rms_val.shape,"\n")

    abs_diff_signal=np.sum(np.abs(np.diff(data,axis=-1)),axis=-1)
    print(Fore.GREEN+"abs_diff_signal\n",abs_diff_signal)
    print(Fore.YELLOW+"Shape =",abs_diff_signal.shape,"\n")

    skewness=stats.skew(data,axis=-1)
    print(Fore.GREEN+"skewness=\n",skewness)
    print(Fore.YELLOW+"Shape =",skewness.shape,"\n")

    Kurtosis=stats.kurtosis(data,axis=-1)
    print(Fore.GREEN+"Kurtosis=\n",Kurtosis)
    print(Fore.YELLOW+"Shape =",Kurtosis.shape,"\n")
    #COncatenate all the Features 
    concatenate_features=np.concatenate((mean,std,ptp,var,minimum,maximum,arg_max,arg_min,rms_val,skewness,Kurtosis,abs_diff_signal),axis=-1)
    #print("concatenate_features Dimension=",concatenate_features.shape)

    features=[]
    for k in range(0,len(data)):
        features.append(concatenate_features[k])
    # convert list to an array
    Features_array=np.array(features)
    print("features_array dimension =",Features_array.shape)
    classifier(Features_array,label_array,group_array)
    pass

def classifier(Features_array,label_array,group_array):
    clf_model=LogisticRegression(max_iter=1000)
    gkf=GroupKFold(10)
    pipe=Pipeline([('scaler',StandardScaler()),('clf',clf_model)])
    param_grid={'clf__C':[0.1,0.5,0.7,1,3,5,7]}
    gscv=GridSearchCV(pipe,param_grid,cv=gkf,n_jobs=12)
    clf_model=gscv.fit(Features_array,label_array,groups=group_array)
    print("Feature array=",Features_array,"\n","Label_array=",label_array)
    y_pred =clf_model.predict(Features_array)
    #print("Accuracy_Score =",gscv.best_score_)
    print("Accuracy: ==",metrics.accuracy_score(label_array,y_pred))
    print("Confusion Matrix: = \n",confusion_matrix(label_array,y_pred),"\n")
    print("Classification Report : = \n",classification_report(label_array, y_pred),"\n")
    plot_confusion_matrix(clf_model, Features_array, label_array)  
    plt.show()
    pass

def main():
    all_file_path=glob('/home/soumen/Desktop/EEG_FEB/dataverse_files/*')
    print("No of Files available= ",len(all_file_path),"\n")
    for i in range (0,len(all_file_path)):
        print(all_file_path[i])
    print("\n")

   # Seperate the healthy_patients and schizophrenia_patients
    healthy_patient_file_path =[i for i in all_file_path if 'h' in i.split('dataverse_files')[1]]
    schizophrenia_patient_file_path =[i for i in all_file_path if 's' in i.split('dataverse_files')[1]]
    print("#Healthy Patients =",len(healthy_patient_file_path),"#schizophrenia_patients=",len(schizophrenia_patient_file_path))

   #See the dimension of one sample data
    sample_data=data_read(healthy_patient_file_path[0])
    print("\n Dimension =",sample_data.shape)   # 231 = No of epochs/segnments , 19 = no of channels ,1250 =length of signals 
   

    healthy_patient_epochs_array=[data_read(i) for i in healthy_patient_file_path]
    schizophrenia_patient_epochs_array=[data_read(i) for i in schizophrenia_patient_file_path]
    epochs_label_annotation( healthy_patient_epochs_array, schizophrenia_patient_epochs_array)
    pass


main()