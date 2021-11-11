# The purpose of this script is to create a code to spatially analyze all the wells in the combined database based on management. 
# Written by Danielle Tadych
# Goals:
# - create columns with the different management regions
#       1. AMA/INA
#       2. Irrigation District
#       3. AI Homelands
#       4. Unregulated

# Potential workflow
# - import master database and shape files
# - Create columns of the different management options
# - Create an if statement of if a well falls within a certain region, then it can equal the name of the shape it falls under
# %%
import os
from geopandas.tools.sjoin import sjoin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime as dt
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp



# %%
# Load in the master database
filename = 'Master_ADWR_database_nocancelled.shp'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
filepath = os.path.join(outputpath, filename)
print(filepath)

masterdb = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb.info())
# %%
# Reading in the shapefile
# GEOREG.to_file('../MergedData/Output_files/Georegions_3col.shp')
filename = "Georegions_fixed.shp"
filepath = os.path.join(outputpath, filename)
georeg = gp.read_file(filepath)

#%%
# Read in the annual time series database
filename = 'Wells55_GWSI_WLTS_DB_annual.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)
annual_db = pd.read_csv(filepath, header=1, index_col=0)
#pd.options.display.float_format = '{:.2f}'.format
annual_db.index.astype('int64')
annual_db.head()

#%%
# Read in the monthly water level time series database
filename = 'Wells55_GWSI_WLTS_DB_monthly.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)
monthly_db = pd.read_csv(filepath, header=1, index_col=0)
#pd.options.display.float_format = '{:.2f}'.format
#monthly_db.index.astype('int64')
monthly_db.head()

#%%
# Read in the yearly Number of measurements database
filename = 'Wells55_GWSI_LEN_TS_DB_annual.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)
lenannual_db = pd.read_csv(filepath, header=1, index_col=0)
#pd.options.display.float_format = '{:.2f}'.format
#monthly_db.index.astype('int64')
lenannual_db.head()

# %% Overlay georegions onto the static database
# Going to use sjoin based off this website: https://geopandas.org/docs/user_guide/mergingdata.html
static_geo = gp.sjoin(masterdb, georeg, how="inner", op='intersects')
static_geo.head()

# %% Create a dataframe of AHS_Region and Well ID's
reg_list = static_geo[['Combo_ID', 'AHS_Region']]
reg_list

# %% Converting Combo_ID to float
reg_list['Combo_ID'] = reg_list['Combo_ID'].astype(int, errors = 'raise')

# %% set indext to REGISTRY_I
reg_list.set_index('Combo_ID', inplace=True)
reg_list

# %%
#annual_db2.REGISTRY_I.astype('int64')
reg_list.reset_index(inplace=True)
reg_list.head()

# %%
#annual_db2 = annual_db.reset_index(inplace=True)
#annual_db2 = annual_db.rename(columns = {'year':'Combo_ID'})
#annual_db2.head()

# %% Switching to monthly
monthly_db2 = monthly_db
# %%
monthly_db2 = monthly_db2.rename(columns = {'date':'Combo_ID'})
monthly_db2.head()

# %% Start filtering the timeseries database to only include November - March
monthly_db2.set_index('Combo_ID', inplace=True)
monthly_db2 = monthly_db2.transpose()
monthly_db2.head()

# %%
monthly_db2.index = pd.to_datetime(monthly_db2.index)

# %%
monthly_db2['month'] = monthly_db2.index.month
print(monthly_db2['month'])

# %%
monthly_db2 = monthly_db2[(monthly_db2['month'] <= 3) +
                            (monthly_db2['month'] >= 11)]
print(monthly_db2)

# Need to delete the month column before 
del monthly_db2['month']
monthly_db2.head()
# %% Reformatting so we can add AHS regions and merge with other lists
monthly_db2 = monthly_db2.transpose()
monthly_db2.head()
# %%
monthly_db3 = monthly_db2.reset_index()
monthly_db3.head()

# %%
monthly_db3['Combo_ID'] = monthly_db3['Combo_ID'].astype(int, errors = 'raise')
monthly_db3['Combo_ID'].head()
# %% 
# Pull out measurements less than 3 in the annual len DB
lenannual_db['sum'] = lenannual_db.sum(axis=1)
print(lenannual_db['sum'])

# %%
len_db = lenannual_db['sum']
len_db

# %%
len_db = len_db[len_db >= 3]
len_db

# %%
len_db2 = len_db.reset_index()
len_db2.info()

# %%
len_db2 = len_db2.rename(columns = {'year':'Combo_ID'})
len_db2.info()
# %%
len_db2['Combo_ID'] = len_db2['Combo_ID'].astype(int, errors = 'raise')
len_db2.info()

# %% Gotta delete the sum column now
del len_db2['sum']
len_db2.info()

# %% Add list of AHS region ID's to the monthly database
combo = pd.merge(monthly_db3,reg_list,how='inner',on='Combo_ID')
#combo2 = combo.drop_duplicates()
combo.info()

# This worked!!
# %%
len_db2 = len_db2.reset_index()
len_db2
# %% Now need to combine len_db 
combo2 = combo.merge(len_db2, how='inner', on='Combo_ID')
#combo3 = pd.merge(combo2,len_db2,how='inner',left_index=True, right_index=True)
combo2.info()
combo2

# %% Now for plotting the timeseries
cat_wl = combo.groupby(['AHS_Region']).mean()
cat_wl

# %% Delete Combo ID column
del cat_wl['Combo_ID']
cat_wl

# %%
cat_wl2 = cat_wl.transpose()
cat_wl2

# %%
cat_wl2.info()

# %% Narrow to only period of analysis
startyear = 1970
endyear = 2020
# %%
cat_wl2.index = pd.to_datetime(cat_wl2.index)
cat_wl2['year'] = cat_wl2.index.year
cat_wl2

# %%
cat_wl2 = cat_wl2[(cat_wl2['year'] >= startyear) +
                            (cat_wl2['year'] >= endyear)]

print(cat_wl2)
#%%
del cat_wl2['year']
cat_wl2.head()

# %% --- Now fun Stats ---
# First get means
wl_mean = cat_wl2.mean()
wl_mean

# %% Subtracting the mean from the values to get anomalies
wl_anomalies = cat_wl2 - wl_mean
wl_anomalies

# %% Going to export all these as CSV's
wl_anomalies.to_csv('../MergedData/Output_files/AGU_Categories_WLanomalies.csv')
combo2.to_csv('../MergedData/Output_files/AGU_WaterLevels.csv')

#%%
cat_wl2.to_csv('../MergedData/Output_files/AGU_WaterLevels_transposedforgraphing.csv')
#%% Plotting the regulated regions
fig, ax = plt.subplots()
ax.plot(wl_anomalies['Reg_CAP'], label = 'Colorado River')
ax.plot(wl_anomalies['Reg_CAP_Other'], label = 'Colorado River & Other SW', color='green')
ax.plot(wl_anomalies['Reg_NoSW'], label = 'No SW', color = 'orange')
ax.set(title='Regulated Water Level Anomalies since 1970', xlabel='Date', ylabel='Water Level Anomaly')
plt.ylim(-300,300)
ax.legend()

# %% Plotting the unregulated regions
fig, ax = plt.subplots()
ax.plot(wl_anomalies['Unreg_CoR'], label = 'Colorado River')
ax.plot(wl_anomalies['Unreg_Other'], label = 'Colorado River & Other SW', color='green')
ax.plot(wl_anomalies['Unreg_NoSW'], label = 'No SW', color = 'orange')
ax.set(title='Unregulated Water Level Anomalies since 1970', xlabel='Date', ylabel='Water Level Anomaly')
plt.ylim(-300,300)
ax.legend()

# %%
