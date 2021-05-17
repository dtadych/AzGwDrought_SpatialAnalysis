
# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

# %%
# Read in the the water  level file
GWSI_folder = 'GWSI_04142020' #GWSI folder name
file_name = 'GWSI_WW_LEVELS.xlsx'
filepath=os.path.join(GWSI_folder, 'Data_Tables', file_name)
print(filepath)

wl_data = pd.read_excel(filepath, parse_dates=['WLWA_MEASUREMENT_DATE'])
print(wl_data.info())

# Rename the columns in wl file to something shorter
wl_data=wl_data.rename(columns={"WLWA_MEASUREMENT_DATE": "date",
                   "WLWA_SITE_WELL_SITE_ID": "wellid",
                   "WLWA_DEPTH_TO_WATER": "depth"}, errors="raise")
# %%
# Read in the file containing basin codes
GWSI_folder = 'GWSI_04142020' #GWSI folder name
file_name = 'GWSI_SITES.xlsx'
filepath=os.path.join(GWSI_folder, 'Data_Tables', file_name)
print(filepath)

basin_data = pd.read_excel(filepath)
print(basin_data.info())

# Rename the columns in basin file to something shorter
basin_data=basin_data.rename(columns={"SITE_WELL_SITE_ID": "wellid",
                   "SITE_ADWBAS_CODE_ENTRY": "basinid"}, errors="raise")

# print number of columns
basin_data['basinid'].nunique()            
# %%
# Merge basin codes and water levels by wellid into new datatable called wl_data2
wl_data2 = wl_data.merge(basin_data, left_on='wellid', right_on='wellid')
print(wl_data2.info())

#%%
# Output wl_data2 to a csv in the specified directory
wl_data2.to_csv('/Users/matthewford/Desktop/Python files/Output_files/wl_data2.csv')

#%%