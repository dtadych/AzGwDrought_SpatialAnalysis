# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

#%%
# Read in pump_wl
filename = 'pump_wl.csv'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files'
filepath = os.path.join(datapath, filename)
print(filepath)

pump_wl = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(pump_wl.info())


# %% ---- HIGH LEVEL SUMMARIES ----
# Makig a pivot table of Basin summary count
Basin_wellcount = pd.pivot_table(pump_wl, index=['Basin'], values='Well Id', aggfunc=np.count_nonzero)
Basin_wellcount
Basin_wellcount.plot(kind='bar')

# Makig a pivot table of Subbasin summary count
Subbasin_wellcount = pd.pivot_table(pump_wl, index=['Subbasin'], values='Well Id', aggfunc=np.count_nonzero)
Subbasin_wellcount
Subbasin_wellcount.plot(kind='bar')

# Makig a pivot table of AMA and INA summary count
AMA_INA_wellcount = pd.pivot_table(pump_wl, index=['AMA INA'], values='Well Id', aggfunc=np.count_nonzero)
AMA_INA_wellcount
AMA_INA_wellcount.plot(kind='bar')

# %% ---- LOW LEVEL SUMMARIES ----
# Filtering for pumping values that are not null
# a quick comparison shows that AF Pumped always has non-zero values
nonzero_pump = pump_wl
nonzero_pump['AFPumped'] = nonzero_pump['AF Pumped']
nonzero_pump_filter = nonzero_pump.AFPumped.notnull()
nonzeropump = nonzero_pump[nonzero_pump_filter]
print(nonzeropump)

# %%
# Switching to Water Level
# Filter for water levels that have data
nonzero_wl = pump_wl
nonzero_wl_filter = nonzero_wl.depth.notnull()
nonzerowl = nonzero_wl[nonzero_wl_filter]
print(nonzerowl)

# %%
# Makig a pivot table of Basin summary count with non-zero wl
Basin_wc_wl = pd.pivot_table(nonzerowl, index=['Basin'], values='Well Id', aggfunc=np.count_nonzero)
print(Basin_wc_wl)
Basin_wc_wl.plot(kind='bar')

# Makig a pivot table of Subbasin summary count
Subbasin_wc_wl = pd.pivot_table(nonzerowl, index=['Subbasin'], values='Well Id', aggfunc=np.count_nonzero)
print(Subbasin_wc_wl)
Subbasin_wc_wl.plot(kind='bar')

# Makig a pivot table of AMA and INA summary count
AMA_INA_wc_wl = pd.pivot_table(nonzerowl, index=['AMA INA'], values='Well Id', aggfunc=np.count_nonzero)
print(AMA_INA_wc_wl)
AMA_INA_wc_wl.plot(kind='bar')

# %%
# ----- Total amount of readings -----
Basin_wi_summary = pd.pivot_table(nonzerowl, index=['Basin'], columns='Well Id', values='depth', aggfunc=np.count_nonzero)
Basin_wi_summary
Basin_wi_summary.plot(kind='bar')

# %%
# ----- Years Available -----
Basin_wi_years_summary = pd.pivot_table(nonzerowl, index=['Subbasin'], values='YEAR')
Basin_wi_years_summary
Basin_wi_years_summary.plot(kind='box')

# %%
# pivot1.to_csv(outputpath + 'AFPumped_Bywellid&date.csv')
