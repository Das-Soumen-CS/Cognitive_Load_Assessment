from cgitb import reset
from json import load
from lib2to3.pgen2.token import LESS
import string
from tokenize import Double
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import sys
import os
import math 
import seaborn as sns
from scipy.stats import skew
from scipy.signal import find_peaks ,peak_prominences ,argrelextrema,peak_widths
import plotly.graph_objects as go

 
time_list=[]

def line_graph_plot(data):
    # resistivity Plot
    data.plot.line(title ="GSR_Responce" ,x ='Time  min:ss',y='GSR Value',figsize=(20,5),grid =True ,subplots=True ,color="green",label = "resistivity")
    plt.legend()
    plt.grid(color = 'purple')
    plt.title("Skin Resistance vs Time ")
    plt.xlabel("Time")
    plt.ylabel("Skin Resistance = kΩ")
    plt.grid(color = 'purple')
    plt.show()
    #Conductivity Plot
    data.plot.line(title ="GSR_Responce" ,x ='Time  min:ss',y='conductivity',figsize=(20,5),grid =True ,subplots=True ,color="blue",label = "conductivity")
    plt.legend() 
    plt.grid(color = 'purple')
    plt.title("Skin Conductance vs Time ")
    plt.xlabel("Time")
    plt.ylabel("Skin Conductance = μS")
    plt.grid(color = 'purple')
    plt.show()
    # Joint Figure Comaprision Resistivity and Conductivity
    plt.plot(data['Time  min:ss'],data['GSR Value'],color="green",label = "Resistivity",linewidth=2.5)
    plt.plot(data['Time  min:ss'],data['conductivity'],color="blue",label = "Conductivity",linewidth=2.5)
    plt.grid(color = 'purple')
    plt.legend()
    plt.title("GSR Responce vs Time ")
    plt.xlabel("Time")
    plt.ylabel("GSR Responce")
    plt.show()
    pass

def histogram_plot(data):
    # Creating dataset
    np.random.seed(23685752)
    N_points = 10000
    n_bins = 20
    # Creating distribution
    #x = data['Time  min:ss']
    x = data['Time_Seconds']
    y = data['GSR Value']
    legend = ['GSR Value Distribution']
    # Creating histogram
    fig, axs = plt.subplots(1, 1,figsize =(10, 7),tight_layout = True)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)
    
    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad = 5)
    axs.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    axs.grid(visible = True, color ='grey',linestyle ='-.', linewidth = 0.5,alpha = 0.6)
    
    # Add Text watermark
    fig.text(0.9, 0.15, 'Id_1',fontsize = 12,color ='red',ha ='right',va ='bottom',alpha = 0.7)
    
    # Creating histogram
    N, bins, patches = axs.hist(x, bins = n_bins)
    
    # Setting color
    fracs = ((N**(1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())
    
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    # Adding extra features   
    plt.xlabel("Time")
    plt.ylabel("GSR Value")
    plt.legend(legend)
    plt.title('GSR vs Time')
    plt.show()

    plt.hist(data['GSR Value'], bins=n_bins,align='right',color="purple",edgecolor='black')
    plt.grid(color = 'green')
    plt.xlabel("Time")
    plt.ylabel("GSR Value")
    plt.legend(legend)
    plt.title('GSR vs Time')
    plt.show()
    pass

def Statistical_Analysis(data):
    gsr_min=data["GSR Value"].min(skipna=True)
    gsr_max=data["GSR Value"].max(skipna=True)
    print("Max GSR/ Max Relax =",gsr_max,"\n")
    print("Min GSR/ Max Load =",gsr_min,"\n")
    gsr_start=data["GSR Value"].iloc[0]
    print("Start GSR/Initial Load =",gsr_start,"\n")
    print("Max load - Initial Load= ", gsr_min-gsr_start,"\n","\n")

    mean_gsr= data["GSR Value"].mean(skipna = False)
    print("Mean GSR = ",mean_gsr,"\n")
    
    median_gsr=data["GSR Value"].median(skipna=False)
    print("Median GSR = ", median_gsr,"\n")

    mode_gsr=data["GSR Value"].mode()
    print("Mode GSR = ",mode_gsr,"\n")
    
    mode_color=data["Color"].mode()
    print("Mode Color =",mode_color,"\n")

    std_gsr=data["GSR Value"].std(skipna=True)
    print("Standard Deviation GSR =",std_gsr,"\n")

    GSR_variance = data['GSR Value'].var()
    print("Variance GSR =",GSR_variance,"\n")

    color_fre=data['Color'].value_counts()
    print("Frequency =\n",color_fre,"\n")

    least_frquent_color = color_fre.index[-1]
    print ("Least frequent color=",least_frquent_color)

    data["GSR_diff"]= data["GSR Value"].diff()
    #print("Append GSR Diff =\n",data)
    data["GSR_diff"]=data["GSR_diff"].fillna(0)
    print("Replace NAN by 0= \n",data)

    #data["Time_diff"]= data["Time Seconds"].diff()
    #print("Append Time_Diff =\n",data,"\n")
    #data["Time_diff"]=data["Time_diff"].fillna(0)
    #print("Replace NAN by 0= \n", data,"\n")

    data['GSR Mean-GSR Value'] = mean_gsr- data['GSR Value']
    print("\nDeviation From Mean = :\n", data)
    print("GSR Mean-Max Load =",data['GSR Mean-GSR Value'].max(),"\n" )
    print("GSR Mean-Initial Load=",data['GSR Mean-GSR Value'].iloc[0] ,"\n")
    # Function Call
    Load_wrt_Mean(mean_gsr,gsr_min)
    Load_wrt_Initial_GSR(gsr_min,gsr_start)
    pass
    #print("Description of data = \n",data.describe())
    
def Load_wrt_Mean(mean_gsr,gsr_min):
    load_score={(mean_gsr-gsr_min)/mean_gsr}  # Here I have used {} => set , need to convert lo list =>then pick first element
    print(type(load_score))
    print('Load Score= ',load_score)
    load_normalized_score=list(load_score)
    print("Load Normalized score wrt Mean_GSR= ",load_normalized_score[0]*100,"\n")
    pass

def Load_wrt_Initial_GSR(Max_load,Initial_gsr):
    load_score=(Max_load-Initial_gsr)/Initial_gsr
    print(type(load_score))
    print('Load Score= ',load_score)
    print("Load Normalized score wrt Initial Load= ",load_score*100,"\n")
    pass
    

def min_sec_to_Seconds(data):
    gsr_time=data['Time  min:ss']
    #print("GSR time =\n ",gsr_time)
    for i in range(0,len(gsr_time)):
        x=gsr_time[i]
        min,sec=divmod(x,1)
        #print(min,"",sec)
        min_sec=min*60
        #print(min_sec)
        Total_sec=min_sec+(sec*100)
        time_list.append(Total_sec)
        #print(Total_sec)
    data["Time_Seconds"]= time_list
    data["Time_Diff"]=data["Time_Seconds"].diff()
    #print("Append Time Diff =\n",data)
    data["Time_Diff"]=data["Time_Diff"].fillna(0)
    print("Replace NAN by 0= \n",data)
    pass

def skewness_Kurtosis(data):
    data.boxplot(by ='GSR_diff', column =['Time_Diff'], grid = False)
    plt.show()
    print("\n Correlation matrix = \n",data.corr())
    result= data.drop(['Color'],axis=1) 
    for i in result:
        plt.figure()
        sns.histplot(result[i],kde=True)
        plt.show()
        pass
    result_skewness=data.skew(axis=0, skipna=True, level=None, numeric_only=True)   
    print("\n Skewness measure = \n")
    print(result_skewness)
    result_kurtosis= data.kurtosis(skipna = True,numeric_only=True)
    print("\n kurtosis measure =\n")
    print(result_kurtosis)
    pass

def peaks_valleys(data):
    time_series = data['GSR Value']
    #Peak Finding
    peaks = argrelextrema(time_series.to_numpy(), np.greater) # local maxima
    peaks = peaks[0]
    print("Peaks are =\n",time_series[peaks],'\n')
    # Vallyes Finding
    valleys_ind = argrelextrema(time_series.to_numpy(), np.less) # Local minima
    valleys_ind = valleys_ind[0]
    print("vallyes are = \n",time_series[valleys_ind],'\n')
    # mean of Peaks
    mean_peaks= time_series[peaks].mean(skipna = False)
    print("Mean Peaks = ",mean_peaks,"\n")
    data['Mean peaks-GSR Value'] = mean_peaks- data['GSR Value']
    print("\nDeviation From Mean Peaks = :\n", data)
    # Mean of Vallyes
    mean_vallyes= time_series[valleys_ind].mean(skipna = False)
    print("Mean vallyes = ",mean_vallyes,"\n")
    data['Mean Vallyes-GSR Value'] = (mean_vallyes- data['GSR Value'])
    print("\n Deviation From Mean Vallyes = :\n", data)
    #Plot figure
    plt.rcParams["figure.figsize"] = (20,5.5)
    plt.plot(time_series,linewidth=2.5,color="black",label="raw GSR data(kΩ)")
    plt.grid(color = 'purple')
    plt.plot(peaks, time_series[peaks], 'o',markersize=10,linestyle='dashed',color='blue',markerfacecolor='red',label="Peak Resistance or local max or SCR") # For Peaks
    plt.plot(valleys_ind, time_series[valleys_ind], 'o',markersize=10,linestyle='dashed',color='red',markerfacecolor='green',label="valley Resistance or local min or SCL")  # For Valleys
    plt.plot(np.zeros_like(time_series), "--", color="yellow",markersize=10,linewidth=2.5)
    # Find Peak prominences
    promin=peak_prominences(time_series,peaks)[0]
    contur_height=time_series[peaks]-promin
    plt.vlines(x=peaks,ymin=contur_height,ymax=time_series[peaks],color="purple",linewidth=2.5)
    # peak_width => for half recovery
    '''
    results_half = peak_widths(time_series, peaks, rel_height=0.5)
    results_half[0]
    plt.hlines(*results_half[1:], color="black",linewidth=2.5,linestyle='dashed')'''
    #peak width for full recovery
    results_full = peak_widths(time_series, peaks, rel_height=1)
    results_full[0]  # widths
    plt.hlines(*results_full[1:], color="green",linewidth=2,linestyle='dashed')
    plt.title("Skin Resistance vs Time ")
    plt.xlabel("Time")
    plt.ylabel("Skin Resistance = kΩ")
    plt.legend()
    plt.show()
    pass


def main():
    path = sys.argv[1]
    sheet_no=sys.argv[2]
    data = pd.read_excel(path,sheet_name= int(sheet_no))
    #print(data)
    data=data.fillna(0)
    print(data)
    data.head(30).boxplot(by='Time  min:ss',column=['GSR Value'], widths=0.5,grid = False,rot=90)
    plt.show()
    data.boxplot(by='Color', grid = False)
    plt.show()
    #Measure Conductivity
    conductivity=1000/data["GSR Value"]
    data["conductivity"]=conductivity
    data["conductivity_diff"]= data["conductivity"].diff().fillna(0)
    # Function call of different functions 
    Statistical_Analysis(data)
    min_sec_to_Seconds(data)
    line_graph_plot(data)
    peaks_valleys(data)
    histogram_plot(data)
    print("\n Description of data = \n",data.describe())
    skewness_Kurtosis(data)
    # Finally Create an Excel Sheet with same file name to the same directory
    prefix=os.path.basename(path)
    prefix=os.path.splitext(prefix)[0]
    if (int(sys.argv[2])==0):
        status="with_out_Task"
        data.to_excel('./'+prefix+status+'_Cognitive_load.xlsx')
    elif(int(sys.argv[2])==1):
        status="with_task"
        data.to_excel('./'+prefix+status+'_Cognitive_load.xlsx')
    else:
        print("Please press either { 0 =for without Task } ,or {1= for with task}")

main()

