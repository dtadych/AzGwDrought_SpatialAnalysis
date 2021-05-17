
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob

# %%
# Read in pump_data_Full file
filepath = os.path.join('/Users/matthewford/Desktop/Python files/Output_files/Pump_Data_Full.csv')
print(filepath)

pump_data_all = pd.read_csv(filepath)
#create not scientific notation 0 decimal places
pd.options.display.float_format = '{:.0f}'.format
print(pump_data_all.info())

#%%
# Read in wl_data2
filepath = os.path.join('/Users/matthewford/Desktop/Python files/Output_files/wl_data2.csv')
print(filepath)

wl_data_all = pd.read_csv(filepath)
# create not scientific notation 2 decimal places
pd.options.display.float_format = '{:.2f}'.format
print(wl_data_all.info())

# %%
# Merge files
pump_wl = wl_data_all.merge(pump_data_all, left_on='wellid', right_on='wellid')
print(pump_wl.info())

#%%
# Output pump_wl to a csv in the specified directory
pump_wl.to_csv('/Users/matthewford/Desktop/Python files/Output_files/Pump_wl.csv')

# %%
