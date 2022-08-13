import pandas as pd
import numpy as np 
import sys
from datetime import datetime
list_1=[]
list_2=[]

def milliseconds_to_day_date_time(data):
    print(type(data))
    data_temp = data[['Flow ID','Task Start Date time','Task End Datetime']]
    print("data =\n",data_temp[:-1],"\n")
    #timestamp=datetime.fromtimestamp(1655619434)
    #print(timestamp.strftime('%Y-%m-%d %H:%M:%S %p'))


    for i in range(len(data_temp)):
        temp_1=data_temp.loc[i,"Task Start Date time"]
        temp_2=data_temp.loc[i,"Task End Datetime"]
        #temp=data_temp.iloc[i,-1] # same thing happens like above line using iloc => ith row ,2nd column
        temp_1=(str(temp_1)[:-3])  # Delete last 3 charecter from date time string
        temp_2=(str(temp_2)[:-3]) 
        timestamp_1=datetime.fromtimestamp(int(temp_1))
        date_time_1=timestamp_1.strftime('%d-%m-%Y %H:%M:%S %p')  # date-month-year hour-min-sec AM/PM
        #print(date_time_1)
        list_1.append(date_time_1) # append all the converted values to the list
        
        timestamp_2=datetime.fromtimestamp(int(temp_2))
        date_time_2=timestamp_2.strftime('%d-%m-%Y %H:%M:%S %p')  # date-month-year hour-min-sec AM/PM
        list_2.append(date_time_2)
    #print(list)         
    #data.insert(5,'Task Start Date_Time',list) # insert in a specific position to a dataframe 
    data['Task Start Date_time']=list_1
    data['Task END Date_time']=list_2
    #data[data_temp.columns.values[1]]=list
    print("data =\n",data,"\n")
    data.to_excel('./Datasheet.xlsx')
 
    
def main():
    file_path=sys.argv[1]
    sheet_no=sys.argv[2]
    data=pd.read_excel(file_path,sheet_name= int(sheet_no))
    print(data)
    milliseconds_to_day_date_time(data)


main()
