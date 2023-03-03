from glob import glob
import os 
import mne
import matplotlib.pyplot as plt
import sys
import numpy as np 
import pandas as pd
from scipy import stats
# To change Font Color
import colorama
from colorama import Fore ,Back, Style
from simple_colors import *
# For  Supervised Classification
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# For CNN Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D,BatchNormalization,LeakyReLU,MaxPool1D,GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session

physical_devices =tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],True)
print("No of GPU  =",len(physical_devices))

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
    print("\n")
    print("length of group list =",len(group_list))

    # convert list to array
    data_array=np.vstack(combined_data_list)
    label_array=np.hstack(combined_label_list)
    group_array=np.hstack(group_list)
    print("\n",data_array.shape,label_array.shape,group_array.shape,"\n")

    # swap (7201, 19, 1250)  ===>(7201,1250,19)  ===> Because kerase wants the "no of channels" at the end
    data_array=np.moveaxis(data_array,1,2)
    print("After swaping Channel position =",data_array.shape,"\n")
   # group_k_fold_cross_validation(data_array,label_array,group_array)
    gkf=GroupKFold()
    for train_index ,val_index in gkf.split(data_array,label_array,groups=group_array):
        train_features ,train_labels=data_array[train_index],label_array[train_index]
        val_features ,val_labels=data_array[val_index],label_array[val_index]
        scaler=StandardScaler()
        train_features=scaler.fit_transform(train_features.reshape(-1,train_features.shape[-1])).reshape(train_features.shape)
        print("train_featutres =",train_features.shape,"\n")
        print("train_features.shape[-1] =",train_features.shape[-1],"\n")
        print("train_features.reshape(-1,train_features.shape[-1])=",train_features.reshape(-1,train_features.shape[-1]).shape,"\n")
        val_featutres=scaler.transform(val_features.reshape(-1,val_features.shape[-1])).reshape(val_features.shape)
    #call CNN model
    evaluate_1D_CNN(train_features,train_labels,val_features,val_labels)
    pass


def evaluate_1D_CNN(train_features,train_labels,val_features,val_labels):
    model=tf.keras.Sequential()
    model.add(Conv1D(filters=5,kernel_size=3, strides=1,input_shape=(1250,19),name='First_Layer'))
    model.add(BatchNormalization())
    model.add(Dense(512,activation = 'LeakyReLU'))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=5,kernel_size=3, strides=1,input_shape=(1250,19),name='Second_Layer'))
    model.add(Dense(512,activation = 'LeakyReLU'))
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Dropout(0.5))

    '''model.add(Conv1D(filters=5,kernel_size=3, strides=1,input_shape=(1250,19),name='Third_Layer'))
    model.add(Dense(256,activation = 'LeakyReLU'))
    model.add(AveragePooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.5))'''

    model.add(Conv1D(filters=5,kernel_size=3, strides=1,input_shape=(1250,19),name='Fourth_Layer'))
    model.add(Dense(256,activation = 'LeakyReLU'))
    model.add(GlobalAveragePooling1D())
    #model.add(Flatten(name='Fifth_Layer'))
    model.add(Dense(1,activation='sigmoid',name='output_Layer')),

    #model=Soumen_CNN_model()
    print("\n")
    print(model.summary())
    model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),   # from_logits=True  beacuse in our model don't have any softmax activation function to the last "Dense" layer
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'],
            )

    # This callback will stop the training when there is no improvement in
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) 
    history=model.fit(train_features,train_labels,epochs=100,batch_size=10,validation_data=(val_features,val_labels),callbacks=[early_stopping])
    #history=model.fit(train_features,train_labels,batch_size=10 ,epochs=10,callbacks=[early_stopping],validation_split=0.1, shuffle=True)
    print(history.history.keys())
    #scores = model.evaluate(val_features,val_labels,batch_size=10)
    print("\n")
    #print("Accuracy=%.2f%%" %(scores[1]*100))
    print("# epochs run =",len(history.history['loss']))
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
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
    print("#Healthy Patients =",len(healthy_patient_file_path),"\n","#schizophrenia_patients=",len(schizophrenia_patient_file_path),"\n")

   #See the dimension of one sample data
    sample_data=data_read(healthy_patient_file_path[0])
    print("\n Dimension =",sample_data.shape)   # 231 = No of epochs/segnments , 19 = no of channels ,1250 =length of signals 
   
   
    healthy_patient_epochs_array=[data_read(i) for i in healthy_patient_file_path]
    schizophrenia_patient_epochs_array=[data_read(i) for i in schizophrenia_patient_file_path]
    # Function call
    epochs_label_annotation( healthy_patient_epochs_array, schizophrenia_patient_epochs_array)
    pass


main()