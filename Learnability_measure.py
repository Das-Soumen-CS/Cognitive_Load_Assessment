import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def learnability(data):
    df=pd.DataFrame(data)
    temp=df.iloc[:,1:]
    #print("Hellooo",temp)
    column_headers = list(temp.columns.values)
    print("The Column Header :", column_headers)
    
  
    for i in range(len(df)+1):
        temp=df.iloc[:,i+1:i+2]  # iterates all rows and column 1 to column 4
        print(" \n",temp)
        column_headers = list(temp.columns.values)
        print("The Column Header :", column_headers)
        line_graph=df.plot.line(title ="Learnability" ,
        x ='Trail No',y=column_headers[:],
        figsize=(15,5),grid =True ,subplots=True ,color="green")
        plt.grid(color = 'purple')
        plt.title("Trial_1_time vs Users ")
        plt.xlabel("Trail_No")
        plt.ylabel("Time")
        plt.show()
        #Compute Avg Task Completion Time
        avg_tct= df[column_headers[:]].mean(skipna = False)
        print("Avg Task Completion ",avg_tct,"\n")

    pass


def main():
    file_path=sys.argv[1]
    sheet_no=sys.argv[2]
    if (sheet_no==str(2)):
        data = pd.read_excel(file_path,sheet_name= int(sheet_no))
        print(data)
        learnability(data)
    else:
        data = pd.read_excel(file_path,sheet_name= int(sheet_no))
        print(data)


    pass


main()