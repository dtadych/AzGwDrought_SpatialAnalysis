# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

#%%
# Read in wl_data2
filename = 'wl_data2.csv'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files'
filepath = os.path.join(datapath, filename)
print(filepath)

wl_data2 = pd.read_csv(filepath, parse_dates=['date'])
pd.options.display.float_format = '{:.2f}'.format
print(wl_data2.info())

# %%
# Read in pump_data_Full file
filename = 'Pump_Data_Full.csv'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files'
filepath = os.path.join(datapath, filename)
print(filepath)

pump_data_all = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(pump_data_all.info())

# %%
wl_summary = wl_data2.pivot_table(values=['depth', 'basinid', 'date', 'SITE_AMA_CODE_ENTRY'],
                                    index='wellid', 
                                    aggfunc={
                                        'depth': len,
                                        'date': [min, max],
                                        'basinid': min,
                                        'SITE_AMA_CODE_ENTRY': min
                                            })

# wl_summary.to_csv(outputpath + 'wellid_wlsummary.csv')

# %%
# GRAPHING
filename = 'Output_fileswellid_wlsummary.csv'
filepath = os.path.join(datapath, filename)
wl_summary_graphing = pd.read_csv(filepath)
print(wl_summary_graphing.info())

# %%
wl_summary_graphing['depth'] = wl_summary['depth'].astype(float)

#%%
wl_summary['SITE_AMA_CODE_ENTRY'] = wl_summary['SITE_AMA_CODE_ENTRY'].astype(str)
wl_summary_AMAINA = wl_summary.pivot_table(values='depth', index='SITE_AMA_CODE_ENTRY', aggfunc=np.sum)

#%%
wl_summary.plt.bar(x='SITE_AMA_CODE_ENTRY', y='depth', title='Number of Well Records in each Basin')

# Renaming AMA Code
# %%
pump_summary = wl_data2.pivot_table(values=['AF Pumped', 'Basin', 'Year', 'AMA_INA'],
                                    index='wellid', 
                                    aggfunc={
                                        'depth': len,
                                        'Year': [min, max],
                                        'basinid': min,
                                        'SITE_AMA_CODE_ENTRY': min
                                            })

# %%
