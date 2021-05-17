
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob

# %%
# Read in casastral id file
GWSI_folder = 'GWSI_04142020' #GWSI folder name
file_name = 'GWSI_SITE_CADASTRAL_HISTORY.xlsx'
filepath=os.path.join(GWSI_folder, 'Data_Tables', file_name)
print(filepath)

#make a datatable called cadastral_data with the read in excel file
cadastral_data = pd.read_excel(filepath)

# Rename some of the columns 
cadastral_data=cadastral_data.rename(columns={"SITE_WELL_SITE_ID": "wellid",
                   "SITE_LOCAL_ID": "cadastralid"}, errors="raise")
print(cadastral_data.info())

#%%
#Read in all pump data csv files from the specified directory 
pump_data_files = sorted(glob('/Users/matthewford/Desktop/Python files/GWSI_04142020/Data_Tables/pump_data_*'))
print(pump_data_files)

#%%
#Concat all those csv files into 1 file
pump_data_all = pd.concat((pd.read_csv(file).assign(filename = file)
    for file in pump_data_files), ignore_index = True)
print(pump_data_all.columns)


# Rename the columns and print the datatable info
pump_data_all=pump_data_all.rename(columns={"CADASTRAL": "cadastralid"}, errors="raise")
print(pump_data_all.info)

# %%
# Merge files pump data file and cadastral id file by cadastral id
# Now we have well id linked to acre feet pumped
pump_data_full = cadastral_data.merge(pump_data_all, left_on='cadastralid', right_on='cadastralid')
print(pump_data_full.columns)

#%%
# Output pump_data_all to a csv in the specified directory
pump_data_full.to_csv('/Users/matthewford/Desktop/Python files/Output_files/Pump_Data_Full.csv')

#%%