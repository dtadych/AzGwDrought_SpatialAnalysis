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
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
import earthpy as et

# %%
# Load in the master database

# Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database.shp')

filename = 'Master_ADWR_database_v2.shp'
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
# %%
annual_db = pd.read_csv(filepath, header=0, index_col=0)
#pd.options.display.float_format = '{:.2f}'.format
annual_db.index.astype('int64')
#%%
annual_db.head()

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
annual_db2 = annual_db.reset_index(inplace=True)
annual_db2 = annual_db.rename(columns = {'year':'Combo_ID'})
annual_db2.head()

# %%
#annual_db2.REGISTRY_I.astype('int64')
reg_list.reset_index(inplace=True)

# %%
reg_list.head()
# %% Add list to the annual database
combo = annual_db2.merge(reg_list, how="outer")
combo.info()

# This worked!!
# %% set index to Combo_ID
combo.set_index('Combo_ID', inplace=True)

# %% Now for plotting the timeseries
cat_wl = combo.groupby(['AHS_Region']).mean()
cat_wl

# %%
cat_wl2 = cat_wl.transpose()

# %% Going to export all these as CSV's
cat_wl.to_csv('../MergedData/Output_files/AHS_Categories_WL.csv')
combo.to_csv('../MergedData/Output_files/AHS_WaterLevels.csv')

#%%
cat_wl2.to_csv('../MergedData/Output_files/AHS_WaterLevels_transposedforgraphing.csv')
#%% Plotting
fig, ax = plt.subplots()
ax.plot(cat_wl2, label=['Regulated - CAP', 'Reglated CAP and Other SW', 'Regulated - No SW', 'Reservations', 'Unregulated - Colorado River', 'Unregulated, No SW', 'Unregulated - Other SW'])
ax.set(title='Average Water Levels since 1853', xlabel='Year', ylabel='Water Level')
ax.legend()

# %%
