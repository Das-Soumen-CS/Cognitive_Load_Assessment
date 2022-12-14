import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def NASA_TLX_Score(data):
    data_temp = data[['Mental Demand Tally','Physical Demand Tally','Temporal Demand Tally','Performance Tally','Effort Tally','Frustration Tally',
    'Mental Demand Score', 'Physical Demand Score','Temporal Demand Score','Performance Score',
    'Effort Score','Frustration Score']]
    #print("data =\n",data_temp[:-1],"\n")
    ''' for i in range(len(data_temp)):
        tally_i=data_temp.iloc[i]
        print("\n")
        print(tally_i) '''

    for i in range(len(data_temp)):
        tally_i=data_temp.iloc[i]
        print("\n")
        print("User_"+str(i)+"_Details===>   ","\n")
        print(tally_i)
        Md_weight=data_temp.loc[i,"Mental Demand Tally"]/15
        Md_score=Md_weight*data_temp.loc[i,"Mental Demand Score"]
        print("\n")
        print("Mental Demand_weight=",Md_weight)
        print("Mental Demand_Score=",Md_score)
        print("\n")

        Pd_weight=data_temp.loc[i,"Physical Demand Tally"]/15   
        pd_score=Pd_weight*data_temp.loc[i,"Physical Demand Score"]
        print("Physical Demand_Weight=",Pd_weight)
        print("Physical Demand_Score=",pd_score)
        print("\n")


        Td_weight=data_temp.loc[i,"Temporal Demand Tally"]/15
        Td_score=Td_weight*data_temp.loc[i,"Temporal Demand Score"]
        print("Temporal Demand_weight=",Td_weight)
        print("Temporal Demand_Score=",Td_score)
        print("\n")

        Per_weight=data_temp.loc[i,"Performance Tally"]/15
        Per_score=Per_weight*data_temp.loc[i,"Performance Score"]
        print("Performance_weight=",Per_weight)
        print("Performance_Score=",Per_score)
        print("\n")

        eff_weight=data_temp.loc[i,"Effort Tally"]/15
        eff_score=eff_weight*data_temp.loc[i,"Effort Score"]
        print("Effort_weight=",eff_weight)
        print("Effort_Score=",eff_score)
        print("\n")

        frus_weight=data_temp.loc[i,"Frustration Tally"]/15
        frus_score=frus_weight*data_temp.loc[i,"Frustration Score"]
        print("Frustration_weight=",frus_weight)
        print("Frustration_Score=",frus_score)
        print("\n")
        pass
        Total_Score = Md_score + pd_score + Td_score + Per_score + eff_score + frus_score
        print("User_"+str(i)+"_over all Task Load Score = ",Total_Score)
    print("\n")
    

    pass    

def main():
    file_path=sys.argv[1]
    sheet_no=sys.argv[2]
    data=pd.read_excel(file_path,sheet_name= int(sheet_no))
    print(data)
    NASA_TLX_Score(data)
    pass

main()