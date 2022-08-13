import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
Sus_usability_score=0


def usabilility_score(rawdata):
    df_temp=rawdata
    No_of_questions=len(df_temp)
    print("No of questions=",No_of_questions )
    print("Row_lenth =",No_of_questions)
    print("Column_Size=",len(df_temp.columns))
    print(df_temp)
    weight=2.5  # Mean(No of questions)/2
    for k in range(1,len(df_temp.columns)):
        Sum_odd=0 
        Sum_even=0  
        X=0
        Y=0
        for i in range(len(df_temp)):
            #print(df_temp.iloc[i,k])
            if(i%2==1):
                Sum_even= Sum_even + df_temp.iloc[i,k]   
            else:
                Sum_odd=Sum_odd + df_temp.iloc[i,k]
            X = Sum_odd - (No_of_questions / 2)
            Y = ((5*No_of_questions)/2) -Sum_even
            Sus_usability_score = (X+Y)*weight

        print("\nSum_even=",Sum_even)
        print("Sum_odd=",Sum_odd)
        print("X=",X,)
        print("Y =",Y)
        print("Sus_usability_Score for ",df_temp.columns[k],"=",Sus_usability_score)


def main():
    file_path=sys.argv[1]
    df=pd.read_excel(file_path)
    usabilility_score(df)
    
    
main()