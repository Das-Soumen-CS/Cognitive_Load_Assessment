import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import openpyxl
import colorama
from colorama import Fore ,Back, Style

def Anxiety_Inventory(data,prefix):
    print(data,"\n")
    print(Fore.YELLOW +"Correlation \n \n",data.corr(),"\n")
    df_trans=data.transpose()
    # Drop the column Qusetionarie => value is not neumeric
    df_trans=df_trans.drop(labels='Qusetionarie',  axis=0)
    print(Fore.CYAN+"After Transpose =\n\n",df_trans,"\n")
    df_trans=df_trans.transpose()
    #print(Fore.GREEN+"After Transpose =\n\n",df_trans,"\n")
    df_dict = df_trans.to_dict()
    print(Fore.YELLOW+"Dictonary_Form=\n\n",df_dict,"\n")
    df_2= pd.DataFrame(df_dict)
    sum_column = df_2.sum(axis=0)
    print(Fore.CYAN+prefix +"_Score"+" =\n")
    print (sum_column,"\n")
    pass


def main():
    file_path=sys.argv[1]
    sheet_no=sys.argv[2]
    data = pd.read_excel(file_path,sheet_name= int(sheet_no))
    wb = openpyxl.load_workbook(file_path)
    prefix=wb.sheetnames[int(sheet_no)]
    Anxiety_Inventory(data,prefix)
    pass

main()
