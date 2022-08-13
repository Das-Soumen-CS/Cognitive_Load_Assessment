from json import load
import string
from tokenize import Double
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import sys

def line_graph_plot(data):
    
    line_graph=data.plot.line(title ="GSR_Responce" ,
    x ='Time  min:ss',y='GSR Value',
    figsize=(20,5),grid =True ,subplots=True ,color="green")
    plt.grid(color = 'purple')
    plt.show()

def histogram_plot(data):
    # Creating dataset
    np.random.seed(23685752)
    N_points = 10000
    n_bins = 20
    
    # Creating distribution
    x = data['Time  min:ss']
    y = data['GSR Value']

    legend = ['GSR Value Distribution']
    
    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)
    
    
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
    axs.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.6)
    
    # Add Text watermark
    fig.text(0.9, 0.15, 'Id_1',
            fontsize = 12,
            color ='red',
            ha ='right',
            va ='bottom',
            alpha = 0.7)
    
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
    print("Append GSR Diff =\n",data)
    data["GSR_diff"]=data["GSR_diff"].fillna(0)
    print("Replace NAN by 0= \n",data)

    data["Time_diff"]= data["Time Seconds"].diff()
    print("Append Time_Diff =\n",data,"\n")
    data["Time_diff"]=data["Time_diff"].fillna(0)
    print("Replace NAN by 0= \n", data,"\n")

    data['GSR Mean-GSR Value'] = mean_gsr- data['GSR Value']
    print("\nDeviation From Mean = :\n", data)
    print("GSR Mean-Max Load =",data['GSR Mean-GSR Value'].max(),"\n" )
    print("GSR Mean-Initial Load=",data['GSR Mean-GSR Value'].iloc[0] ,"\n")
    # Function Call
    Load_wrt_Mean(mean_gsr,gsr_min)
    Load_wrt_Initial_GSR(gsr_min,gsr_start)

    print("Description of data = \n",data.describe())
    


def Load_wrt_Mean(mean_gsr,gsr_min):
    load_score={(mean_gsr-gsr_min)/mean_gsr}  # Here I have used {} => set , need to convert lo list =>then pick first element
    print(type(load_score))
    print('Load Score= ',load_score)
    load_normalized_score=list(load_score)
    print("Load Normalized score wrt Mean_GSR= ",load_normalized_score[0]*100,"\n")

def Load_wrt_Initial_GSR(Max_load,Initial_gsr):
    load_score=(Max_load-Initial_gsr)/Initial_gsr
    print(type(load_score))
    print('Load Score= ',load_score)
    print("Load Normalized score wrt Initial Load= ",load_score*100,"\n")
    
    


def main():
    path = sys.argv[1]
    sheet_no=sys.argv[2]
    data = pd.read_excel(path,sheet_name= int(sheet_no))
    print(data)
    data=data.fillna(0)
    print(data)
    line_graph_plot(data)
    histogram_plot(data)
    Statistical_Analysis(data)
   


main()

