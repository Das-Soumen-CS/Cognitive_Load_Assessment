import pandas as pd
import numpy as np
import sys 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.signal import find_peaks ,peak_prominences ,argrelextrema,peak_widths
import scipy.stats as stats

def intrinsic_cog_load(data_final):
    data_final = data_final[data_final['BaseLine Deviation'] > 0]
    print('\n Data after BaseLine filtering =:\n',data_final)
    data_final=data_final.loc[data_final['Color'].isin(['yellow', 'R1', 'R2','R3','R4','R5'])& (data_final['Rank_1'] <len(data_final['Rank_1']))]
    print('\n Data consisting moderate/high load :\n', data_final)
    # Filter data where time >= 15.00 min
    data_final.loc[:,:]=data_final.query("`Time  min:ss` >= 15.00 ")
    #data_final=temp_df  
    #Re_Rank The Filter Data
    data_final["Rank_1"] = data_final["Normalized_Score_1"].rank(ascending=False)
    data_final["Rank_2"] = data_final["Normalized_Score_2"].rank(ascending=True)
    data_final["Rank_3"] = data_final["Normalized_Score_3"].rank(ascending=False)
    data_final["Rank_4"] = data_final["BaseLine Deviation"].rank(ascending=False)
    #print("After Re_ranking =\n",data_final)
    # Filter data where color range R1-R5 ,exclude yellow
    data_final=data_final.loc[data_final['Color'].isin([ 'R1', 'R2','R3','R4','R5'])& (data_final['Rank_1'] <len(data_final['Rank_1']))]
    #print('\n Data consisting high load :\n', data_final.iloc[ :,4:] )
    df=data_final.drop(['GSR Value', 'conductivity'], axis=1)
    print('\n Data consisting high load :\n', df )
    print("\n Current Dimension=",data_final.shape)
    pass

def peaks_valleys(data):
    normalized_score(data)
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
    mean_peaks_conductivity=1000/mean_peaks
    print("mean_peaks_conductivity=",mean_peaks_conductivity)
    #data['Mean peak'] = mean_peaks
    data['Mean peaks-GSR Value'] = mean_peaks- data['GSR Value']
    print("\nDeviation From Mean Peaks = :\n", data)
    # Mean of Vallyes
    mean_vallyes= time_series[valleys_ind].mean(skipna = False)
    print("Mean vallyes = ",mean_vallyes,"\n")
    mean_vallyes_conductivity=1000/mean_vallyes
    print("mean_vallyes_conductivity=",mean_vallyes_conductivity)
    #data['Mean vallyes'] = mean_vallyes
    data['Mean Vallyes-GSR Value'] = (mean_vallyes- data['GSR Value'])
    print("\n Deviation From Mean Vallyes = :\n", data)

    data['BaseLine Deviation'] = ((1000/data["GSR Value"])-mean_vallyes_conductivity)
    data["Rank_4"] = data["BaseLine Deviation"].rank(ascending=False)
    print("\n BaseLine Deviation= :\n", data)
   

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
    #sns.heatmap(data[['Mean peaks-GSR Value','Mean Vallyes-GSR Value','GSR Value']].corr(), cmap='RdBu',vmin=-1, vmax=1, linewidth=0.5, annot=True,annot_kws={'fontsize':11, 'fontweight':'bold'},square=True)
    intrinsic_cog_load(data)
    pass


def normalized_score(data):
    #Measure Conductivity
    conductivity=1000/data["GSR Value"]
    data["conductivity"]=conductivity
    mean_conductivity= data["conductivity"].mean(skipna = False)
    #print(data)
    print("Mean conductivity = ",mean_conductivity,"\n")
    std_gsr=data["conductivity"].std(skipna=True)
    print("Standard Deviation of Conductivity =",std_gsr,"\n")
    #Normalization version 1
    zscores = stats.zscore(conductivity)
    data["Normalized_Score_1"]=zscores
    data["Rank_1"] = data["Normalized_Score_1"].rank(ascending=False)
    # Normalization_version_2
    tscores=50+ 10*zscores
    data["Normalized_Score_2"]=tscores
    data["Rank_2"] = data["Normalized_Score_2"].rank(ascending=True)
    #Normalization version 3 
    norm_3=(conductivity - mean_conductivity)-zscores
    data["Normalized_Score_3"]=norm_3
    data["Rank_3"] = data["Normalized_Score_3"].rank(ascending=False)
    # Normalization_version_4
    print(data)
    pass


def main():
    start = time.time()
    path = sys.argv[1]
    sheet_no=sys.argv[2]
    data = pd.read_excel(path,sheet_name= int(sheet_no))
    #print(data)
    data=data.fillna(0)
    print(data)
    peaks_valleys(data)
    #normalized_score(data)
    time.sleep(1)
    end = time.time()
    print("\n")
    print(f"Runtime of the program is ={end - start}")
    pass


main()