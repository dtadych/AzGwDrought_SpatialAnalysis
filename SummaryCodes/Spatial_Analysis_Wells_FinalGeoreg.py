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
from cProfile import label
from operator import ge
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
#import earthpy as et
import scipy.stats as sp
# %%
# Load in the master database

# Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database.shp')
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/Final_Georegions/'
# %%
filename_mdb_nd = 'Master_ADWR_database_noduplicates.shp'
filepath = os.path.join(outputpath, filename_mdb_nd)
print(filepath)

masterdb = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb.info())

# %%
filename_mdb_w = 'Master_ADWR_database_water.shp'
filepath = os.path.join(outputpath, filename_mdb_w)
print(filepath)

masterdb_water = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb_water.info())
# %%
# Reading in the shapefile
# GEOREG.to_file('../MergedData/Output_files/Georegions_3col.shp')
filename_georeg = "Final_Georegions.shp"
filepath = os.path.join(shapepath, filename_georeg)
georeg = gp.read_file(filepath)
# %%
#georeg.boundary.plot()
georeg.plot(cmap='viridis')

#%%
georeg['GEOREGI_NU'] = georeg['GEOREGI_NU'].astype('int64')
georeg.info()
#%%
# Read in the annual time series database
filename_ts = 'Wells55_GWSI_WLTS_DB_annual.csv'
filepath = os.path.join(outputpath, filename_ts)
print(filepath)
annual_db = pd.read_csv(filepath, header=1, index_col=0)
annual_db.head()
#pd.options.display.float_format = '{:.2f}'.format
#%%
annual_db.index.astype('int64')
#%%
annual_db.head()

# %% Overlay georegions onto the static database
# Going to use sjoin based off this website: https://geopandas.org/docs/user_guide/mergingdata.html
print("Non-cancelled: ", masterdb.crs, "Water Wells: ", masterdb_water.crs, "Georegions: ", georeg.crs)

# %%
georeg = georeg.to_crs(epsg=26912)
# %%
static_geo = gp.sjoin(masterdb, georeg, how="inner", op='intersects')
static_geo.head()
print(str(filename_mdb_nd) + " and " + str(filename_georeg) + " join complete.")

# %% Exporting it because I guess I did that before since I load it in later
# static_geo.to_csv('../MergedData/Output_files/Final_Static_geodatabase_allwells.csv')

# %% Create a dataframe of Final_Region and Well ID's
reg_list = static_geo[['Combo_ID', 'GEO_Region', 'GEOREGI_NU','Water_CAT', 'Loc','Regulation']]
reg_list

# %% Converting Combo_ID to int
reg_list['Combo_ID'] = reg_list['Combo_ID'].astype(int, errors = 'raise')

# %%
annual_db2 = annual_db.reset_index(inplace=True)
annual_db2 = annual_db.rename(columns = {'year':'Combo_ID'})
annual_db2.head()

# %% Add list to the annual database
combo = annual_db2.merge(reg_list, how="outer")
combo.info()

# This worked!!
# %% set index to Combo_ID
combo.set_index('Combo_ID', inplace=True)

# %% Sort the values
combo = combo.sort_values(by=['GEOREGI_NU'])
combo

# %% Now for aggregating by category for the timeseries
cat_wl = combo.groupby(['GEO_Region', 'GEOREGI_NU']).mean()
#cat_wl = combo.groupby(['GEOREGI_NU']).mean()
cat_wl

# %%
cat_wl2 = cat_wl.sort_values(by=['GEOREGI_NU'])
cat_wl2

# %% 
cat_wl2 = cat_wl2.reset_index()
cat_wl2
# %%
#del cat_wl2['GEOREGI_NU']
del cat_wl2['GEO_Region']

# %%
cat_wl2 = cat_wl2.set_index("GEOREGI_NU")
# %%
cat_wl2 = cat_wl2.transpose()
cat_wl2.info()

# %% Trying to fix the year issue
cat_wl2.reset_index(inplace=True)
cat_wl2.info()
# %%
cat_wl2['index'] = pd.to_numeric(cat_wl2['index'])
cat_wl2.info()
# %%
cat_wl2['index'] = cat_wl2['index'].astype(int)
# %%
cat_wl2.set_index('index', inplace=True)
cat_wl2.info()

# %% Going to export all these as CSV's
#cat_wl.to_csv('../MergedData/Output_files/Final_Categories_WL_adjusted.csv')
#combo.to_csv('../MergedData/Output_files/Final_WaterLevels_adjusted.csv')
#cat_wl2.to_csv('../MergedData/Output_files/Final_WaterLevels_transposedforgraphing_allwells_adjusted.csv')

# %% Creating dictionary of labels
#labels = cat_wl2.columns.tolist()
georeg = georeg.sort_values(by=['GEOREGI_NU'])
labels = dict(zip(georeg.GEOREGI_NU, georeg.GEO_Region))
labels

# %% Creating colors
c_1 = '#8d5a99'
c_2 = "#d7191c"
c_3 = '#e77a47'
c_4 = '#2cbe21'
c_5 = '#2f8c73'
c_6 = '#6db7e8'
c_7 = '#165782'
c_8 = '#229ce8'
c_9 = '#1f78b4'
c_10 = '#41bf9e'
c_11 = '#7adec4'
drought_color = '#ffa6b8'
wet_color = '#b8d3f2'

#%% Plotting
ds = cat_wl2
minyear=1970
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 400
fsize = 14

# Plot all of them on a single graph
fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds[2.0], label='Regulated with CAP', color=c_2) 
ax.plot(ds[3.0], label='Regulated without CAP', color=c_3) 
ax.plot(ds[4.0], color=c_4, label='Lower Colorado River - SW Dominated')
ax.plot(ds[5.0], color=c_5, label='Upper Colorado River - Mixed')
ax.plot(ds[10.0], color=c_10, label='North - Mixed')
ax.plot(ds[11.0], color=c_11, label='Central - Mixed')
ax.plot(ds[7.0], color=c_7, label='Northwest - GW Dominated')
ax.plot(ds[9.0], color=c_9, label='Northeast - GW Dominated')
ax.plot(ds[8.0], color=c_8, label='South central - GW Dominated')
ax.plot(ds[6.0], color=c_6, label='Southeast - GW Dominated')
# Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)

# Wet years (2005 and 2010)
g = 2005
h = 2010
ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
ax.grid(True)
ax.set(title=name, xlabel='Year', ylabel='Water Level (ft)')
ax.legend(loc = [1.04, 0.40])

#%% Plot just the regulated
fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds[2.0], label='Regulated with CAP', color=c_2) 
ax.plot(ds[3.0], label='Regulated without CAP', color=c_3) 
ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
ax.grid(True)
ax.set(title=name, xlabel='Year', ylabel='Water Level (ft)')
ax.legend(loc = [1.04, 0.40])
# Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999
plt.axvspan(a, b, color='#ffa6b8', alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color='#ffa6b8', alpha=0.5, lw=0)
plt.axvspan(e, f, color='#ffa6b8', alpha=0.5, lw=0)
# Wet years (2005 and 2010)
g = 2005
h = 2010
ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

#%% Plot SW Dominated and Mixed
fig, ax = plt.subplots(figsize = (16,9))
ax.plot(ds[4.0], color=c_4, label='Lower Colorado River - SW Dominated')
ax.plot(ds[5.0], color=c_5, label='Upper Colorado River - Mixed')
ax.plot(ds[10.0], color=c_10, label='North - Mixed')
ax.plot(ds[11.0], color=c_11, label='Central - Mixed')
ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
ax.grid(True)
ax.set(title=name, xlabel='Year', ylabel='Water Level (ft)')
ax.legend(loc = [1.04, 0.40])

# Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999
plt.axvspan(a, b, color='#ffa6b8', alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color='#ffa6b8', alpha=0.5, lw=0)
plt.axvspan(e, f, color='#ffa6b8', alpha=0.5, lw=0)
# Wet years (2005 and 2010)
g = 2005
h = 2010
ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

# %%
# Plot just the Groundwater Dominated
fig, ax = plt.subplots(figsize = (16,9))
ax.plot(ds[7.0], color=c_7, label='Northwest - GW Dominated')
ax.plot(ds[9.0], color=c_9, label='Northeast - GW Dominated')
ax.plot(ds[8.0], color=c_8, label='South central - GW Dominated')
ax.plot(ds[6.0], color=c_6, label='Southeast - GW Dominated')
ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
ax.grid(True)
ax.set(title=name, xlabel='Year', ylabel='Water Level (ft)')
ax.legend(loc = [1.04, 0.40])

# Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999
plt.axvspan(a, b, color='#ffa6b8', alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color='#ffa6b8', alpha=0.5, lw=0)
plt.axvspan(e, f, color='#ffa6b8', alpha=0.5, lw=0)
# Wet years (2005 and 2010)
g = 2005
h = 2010
ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

# %% Plot in a four panel graph
ds = cat_wl2
minyear=1970
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 400
fsize = 14
ylabel = "Water Level (ft)"

# del ds.at[2015, 10]
ds.at[2015, 10] = None

# For the actual figure
fig, ax = plt.subplots(2,2,figsize=(18,10))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.09)
fig.supxlabel("Year", fontsize = 14, y=0.08)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0,0].plot(ds[2], label='Regulated with CAP', color=c_2) 
ax[0,0].plot(ds[3], label='Regulated without CAP', color=c_3) 
ax[1,1].plot(ds[4], color=c_4, label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds[5], color=c_5, label='Upper Colorado River - Mixed')
ax[0,1].plot(ds[10], color=c_10, label='North - Mixed')
ax[0,1].plot(ds[11], color=c_11, label='Central - Mixed')
ax[1,0].plot(ds[7], color=c_7, label='Northwest - GW Dominated')
ax[1,0].plot(ds[9], color=c_9, label='Northeast - GW Dominated')
ax[1,0].plot(ds[8], color=c_8, label='South central - GW Dominated')
ax[1,0].plot(ds[6], color=c_6, label='Southeast - GW Dominated')

# ax.plot(ds[2.0], label='Regulated with CAP', color=c_2) 
# ax.plot(ds[3.0], label='Regulated without CAP', color=c_3) 
# ax.plot(ds[4.0], color=c_4, label='Lower Colorado River - SW Dominated')
# ax.plot(ds[5.0], color=c_5, label='Upper Colorado River - Mixed')
# ax.plot(ds[10.0], color=c_10, label='North - Mixed')
# ax.plot(ds[11.0], color=c_11, label='Central - Mixed')
# ax.plot(ds[7.0], color=c_7, label='Northwest - GW Dominated')
# ax.plot(ds[9.0], color=c_9, label='Northeast - GW Dominated')
# ax.plot(ds[8.0], color=c_8, label='South central - GW Dominated')
# ax.plot(ds[6.0], color=c_6, label='Southeast - GW Dominated')

ax[0,0].set_xlim(minyear,maxyear)
ax[0,1].set_xlim(minyear,maxyear)
ax[1,0].set_xlim(minyear,maxyear)
ax[1,1].set_xlim(minyear,maxyear)
ax[0,0].set_ylim(max_y,min_y)
ax[0,1].set_ylim(max_y,min_y)
ax[1,0].set_ylim(max_y,min_y)
ax[1,1].set_ylim(max_y,min_y)
ax[0,0].grid(True)
ax[0,1].grid(True)
ax[1,0].grid(True)
ax[1,1].grid(True)
#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)

# # Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999

ax[0,0].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[0,0].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[0,0].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax[1,0].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[1,0].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[1,0].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax[0,1].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[0,1].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[0,1].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax[1,1].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[1,1].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[1,1].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)

# Wet years (2005 and 2010)
g = 2005
h = 2010
ax[0,0].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[0,0].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax[0,1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[0,1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax[1,0].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[1,0].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax[1,1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[1,1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

# ax[0,0].legend(loc = [0.1, 0.15], fontsize = fsize)
# ax[0,1].legend(loc = [0.1, 0.05], fontsize = fsize)
# ax[1,0].legend(loc = [0.1, 0.05], fontsize = fsize)
# ax[1,1].legend(loc = [0.1, 0.20], fontsize = fsize)

# plt.savefig(outputpath+name+'_4panel')
# plt.savefig(outputpath+name+'_4panel_drought')


# %% Plot in a three panel graph
ds = cat_wl2
minyear=2002
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 400
fsize = 14
ylabel = "Water Level (ft)"

# del ds.at[2015, 10]
ds.at[2015, 10] = None

# For the actual figure
fig, ax = plt.subplots(3,1,figsize=(15,15))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.09)
fig.supxlabel("Year", fontsize = 14, y=0.08)
# Panel numbers for easy adjusting
p1 = 0 # Panel 1
p2 = 1 # Panel 2
p3 = 2 # Panel 3
p4 = 3 # Panel 4
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[p1].plot(ds[2], label='Regulated with CAP', color=c_2) 
ax[p3].plot(ds[3], label='Regulated without CAP', color=c_3) 
ax[p1].plot(ds[4], color=c_4, label='Lower Colorado River - SW Dominated')
ax[p2].plot(ds[5], color=c_5, label='Upper Colorado River - Mixed')
ax[p2].plot(ds[10], color=c_10, label='North - Mixed')
ax[p2].plot(ds[11], color=c_11, label='Central - Mixed')
ax[p3].plot(ds[7], color=c_7, label='Northwest - GW Dominated')
ax[p3].plot(ds[9], color=c_9, label='Northeast - GW Dominated')
ax[p3].plot(ds[8], color=c_8, label='South central - GW Dominated')
ax[p3].plot(ds[6], color=c_6, label='Southeast - GW Dominated')

ax[p1].set_xlim(minyear,maxyear)
ax[p2].set_xlim(minyear,maxyear)
ax[p3].set_xlim(minyear,maxyear)
# ax[p4].set_xlim(minyear,maxyear)
ax[p1].set_ylim(max_y,min_y)
ax[p2].set_ylim(max_y,min_y)
ax[p3].set_ylim(max_y,min_y)
# ax[p4].set_ylim(max_y,min_y)
ax[p1].grid(True)
ax[p2].grid(True)
ax[p3].grid(True)
# ax[p4].grid(True)
#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)

# # Drought Year Shading
# a = 2011
# b = 2015.999
# c = 2018.001
# d = 2018.999
# e = 2006
# f = 2007.999

# ax[0,0].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[0,0].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[0,0].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# ax[1,0].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[1,0].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[1,0].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# ax[0,1].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[0,1].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[0,1].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# ax[1,1].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[1,1].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[1,1].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)

# # Wet years (2005 and 2010)
# g = 2005
# h = 2010
# ax[0,0].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[0,0].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[0,1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[0,1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[1,0].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[1,0].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[1,1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[1,1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

ax[p1].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p2].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p3].legend(loc = [1.02, 0.30], fontsize = fsize)
# ax[p4].legend(loc = [0.1, 0.20], fontsize = fsize)

plt.savefig(outputpath+name+'_3panel', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
# plt.savefig(outputpath+name+'_3panel_drought')
# %% --- Now making graphs for other things
# - Well Density (cumulative number of wells) over time
# - max screen depth over time (Casing_DEP vs Installed)
# - Number of new wells installed over time

# Re-read in after proper install date formatting
#       Note: had to go into excel and change the date from mm/dd/yy to mm/dd/yyyy
#             because date parser cutoff for older items is 1969
filename = 'Final_Static_geodatabase_waterwells.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)
static_geo2 = pd.read_csv(filepath 
                          ,parse_dates=['INSTALLED']
                          )
static_geo2

# %% 
#static_geo2 = static_geo
#static_geo2.info()

# %%
#static_geo2['APPROVED'] = pd.to_datetime(static_geo2['APPROVED'])
#static_geo2['APPROVED'].describe()
# %% Only run this if date parser didn't work
static_geo2['INSTALLED'] = pd.to_datetime(static_geo2['INSTALLED'])
static_geo2['INSTALLED'].describe()
# %%
static_geo2['In_year'] = static_geo2['INSTALLED'].dt.year
static_geo2['In_year'].describe()

# %% 
Well_Depth = static_geo2[['WELL_DEPTH', 'INSTALLED', 'Combo_ID', 'In_year','GEOREGI_NU']]
#%%
Well_Depth

#%%
# Well_Depth.to_csv('../MergedData/Output_files/Final_WellDepth.csv')
#%%
Well_Depth = pd.pivot_table(static_geo2, index=["In_year"], columns=["GEOREGI_NU"], values=["WELL_DEPTH"], dropna=False, aggfunc=np.mean)
Well_Depth.describe()

# %% Set shallow and drilling depths
shallow = 200
deep = 500

# %%
wd1 = static_geo2[(static_geo2["WELL_DEPTH"] > deep)]
wd2 = static_geo2[(static_geo2["WELL_DEPTH"] <= deep) & (static_geo2["WELL_DEPTH"] >= shallow)]
wd3 = static_geo2[(static_geo2["WELL_DEPTH"] < shallow)]

# %%
wd1 = wd1.sort_values(by=['GEOREGI_NU'])
wd2 = wd2.sort_values(by=['GEOREGI_NU'])
wd3 = wd3.sort_values(by=['GEOREGI_NU'])
#%%
wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["GEOREGI_NU"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["GEOREGI_NU"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["GEOREGI_NU"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

#wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
#wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
#wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

#%% Exporting the Depth categories
#wdc1.to_csv('../MergedData/Output_files/Final_Welldepth' + str(deep) + 'plus.csv')
#wdc2.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'to' + str(deep) + '.csv')
#wdc3.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'minus.csv')

# %% Plotting fun
# Plot all of the deep wells
ds = wdc1
name = 'Number of Deep Wells (greater than '+ str(deep) +' ft)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(3,figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
#fig.supylabel(ylabel, fontsize = 14, x=0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[0].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[1].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[1].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
ax[1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[1].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[2].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[2].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[2].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[2].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[0].legend(loc = [1.05, 0.40], fontsize = fsize)
ax[1].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

plt.savefig(outputpath+name)

# %% Mid-range wells
# Plot all of them
ds = wdc2
name = 'Number of Mid-range Wells (between '+ str(shallow) +' and '+ str(deep)+' ft.)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(3,figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
#fig.supylabel(ylabel, fontsize = 14, x=0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[0].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[1].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[1].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
ax[1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[1].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[2].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[2].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[2].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[2].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[0].legend(loc = [1.05, 0.40], fontsize = fsize)
ax[1].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

plt.savefig(outputpath+name)

# %% Shallow wells
ds = wdc3
name = 'Number of Shallow Wells (less than '+ str(shallow) +' ft)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(3,figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
#fig.supylabel(ylabel, fontsize = 14, x=0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[0].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[1].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[1].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
ax[1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[1].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[2].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[2].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[2].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[2].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[0].legend(loc = [1.05, 0.40], fontsize = fsize)
ax[1].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

plt.savefig(outputpath+name)

# %%
new_wells = pd.pivot_table(static_geo2, index=["In_year"], columns=["GEOREGI_NU"], values=["INSTALLED"], dropna=False, aggfunc=len)
new_wells
# %%
new_wells.to_csv('../MergedData/Output_files/Final_NewWells.csv')

# %%
ds = new_wells
name = 'Number of new wells per region'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
min_y = 0
max_y = 100
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(3,figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
#fig.supylabel(ylabel, fontsize = 14, x=0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[0].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[1].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[1].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
ax[1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[1].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[2].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[2].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[2].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[2].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[0].legend(loc = [1.05, 0.40], fontsize = fsize)
ax[1].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

plt.savefig(outputpath+name)

# %% ---- Fancier Analyses ----
# Calculating well densities
new_wells2 = pd.read_csv('../MergedData/Output_files/Final_NewWells.csv',
                        header=1, index_col=0)
new_wells2 = new_wells2.reset_index()
new_wells2 = new_wells2.iloc[1:, :]
#new_wells2 = new_wells2.rename(columns = {'GEOREGI_NU':'Year'})
new_wells2 = new_wells2.set_index('GEOREGI_NU')
new_wells2

# %% Calculate the region area
# tost["area"] = tost['geometry'].area/ 10**6
georeg['area'] = georeg.geometry.area/10**6
georeg
# %%
georeg2 = pd.DataFrame(georeg)
georeg2

# %%
#del georeg2['geometry']
#georeg2.info()
# %%
#georeg2.to_csv('../MergedData/Output_files/georegions_area.csv')

#%%
georeg_area = georeg2[['GEOREGI_NU','area']]
georeg_area.info()

# %%
georeg_area = georeg_area.set_index('GEOREGI_NU')
georeg_area = georeg_area.transpose()
georeg_area

# %%
new_wells2 = new_wells.reset_index()
new_wells2
# %% df1.div(df2.iloc[0], axis='columns') - This only returns nan
well_densities = new_wells2.div(georeg_area.iloc[0])
well_densities

# %% df1/df2.values[0,:]
well_densities = new_wells2/georeg_area.values[0,:]
well_densities

# %%
well_densities['1'].plot()

# %%
well_densities.info()

# %%
well_densities1 = well_densities.reset_index()
well_densities1.info()

# %%
well_densities1['GEOREGI_NU'] = pd.to_numeric(well_densities1['GEOREGI_NU'])
well_densities1.info()

# %%
well_densities1['GEOREGI_NU'] = well_densities1['GEOREGI_NU'].astype(int)
well_densities1.info()

# %%
well_densities1.set_index('GEOREGI_NU', inplace=True)
well_densities1.info()

# %%
ds = well_densities1
name = 'Well Densities Per region (#/km2)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
min_y = 0
max_y = 10e-8
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(3,figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
#fig.supylabel(ylabel, fontsize = 14, x=0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0].plot(ds['2'], label='Regulated with CAP', color=c_2) 
ax[0].plot(ds['3'], label='Regulated without CAP', color=c_3) 
ax[1].plot(ds['4'], color=c_4, label='Lower Colorado River - SW Dominated')
ax[1].plot(ds['5'], color=c_5, label='Upper Colorado River - Mixed')
ax[1].plot(ds['10'], color=c_10, label='North - Mixed')
ax[1].plot(ds['11'], color=c_11, label='Central - Mixed')
ax[2].plot(ds['7'], color=c_7, label='Northwest - GW Dominated')
ax[2].plot(ds['9'], color=c_9, label='Northeast - GW Dominated')
ax[2].plot(ds['8'], color=c_8, label='South central - GW Dominated')
ax[2].plot(ds['6'], color=c_6, label='Southeast - GW Dominated')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[0].legend(loc = [1.05, 0.40], fontsize = fsize)
ax[1].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

plt.savefig(outputpath+name+"fixed_axes")

# %% Trying to plot other charts - below here is mostly garbage
georeg = georeg.sort_values(by=['GEOREGI_NU'])
labels = dict(zip(georeg.GEOREGI_NU, georeg.GEO_Region))
labels

barcolors = [c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11]
# %% Kinda worked!  Ned to figure out how to specify things a little more
ds = well_densities1
plt.figure(figsize = (9,6))
plt.bar(ds.columns, ds.sum(), color = barcolors)

# %% fail
ds = new_wells
plt.figure(figsize = (9,6))
plt.pie(ds.columns, ds.sum(), colors=barcolors)

# %% fail
ds = well_densities1
plt.figure(figsize = (9,6))
plt.boxplot(ds.sum())

# %%
stats = pd.DataFrame(index=['slp','int','r_sq','p_val','std_er'],columns=ds.columns)
print(stats)

# %%
stats = pd.DataFrame()
# %% -- Linear regression --
# Actual documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# Tutorial from https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/
#y=np.array(df['OW2 As(mg/L)'].dropna().values, dtype=float)
#x=np.array(pd.to_datetime(df['OW2 As(mg/L)'].dropna()).index.values, dtype=float)
#slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
ds = cat_wl2
min_yr = 2002
mx_yr = 2020
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression:"
print(Name)

#f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]

# -- For Multiple years --
Name = "Linear Regression for Non-drought years: "
wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
#f = ds[(ds.index == wetyrs)]

f = pd.DataFrame()
for i in dryyrs:
        wut = ds[(ds.index == i)]
        f = f.append(wut)
print(f)

stats = pd.DataFrame()
for i in range(1, 12, 1):
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
#        print('Georegion Number: ', i, '\n', 
#                'slope = ', slope, '\n', 
#                'intercept = ', intercept, '\n', 
#                'r^2 = ', r_value, '\n', 
#                'p-value = ', p_value, '\n', 
#                'std error = ', std_err)
        
        # row1 = pd.DataFrame([slope], index=[i], columns=['slope'])
        # row2 = pd.DataFrame([intercept], index=[i], columns=['intercept'])
        # stats = stats.append(row1)
        # stats = stats.append(row2)
        # stats['intercept'] = intercept
        stats = stats.append({'slope': slope, 
                        #       'int':intercept, 
                              'rsq':r_value, 
                              'p_val':p_value, 
                              'std_err':std_err}, 
                              ignore_index=True)
        xf = np.linspace(min(x),max(x),100)
        xf1 = xf.copy()
        #xf1 = pd.to_datetime(xf1)
        yf = (slope*xf)+intercept
        fig, ax = plt.subplots(1, 1)
        ax.plot(xf1, yf,label='Linear fit', lw=3)
        df.plot(ax=ax,marker='o', ls='')
        ax.set_ylim(max(y),0)
        ax.legend()


# stats = stats.append(slope)
#        stats[i] = stats[i].append(slope)

#   df = df.append({'A': i}, ignore_index=True)
stats.index = labels.values()
stats1 = stats.transpose()
del stats1['Reservation']
stats1

# %% Data visualization
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
#xf1 = pd.to_datetime(xf1)
yf = (slope*xf)+intercept
f, ax = plt.subplots(1, 1)
ax.plot(xf1, yf,label='Linear fit', lw=3)
ds.plot(ax=ax,marker='o', ls='')
ax.legend();
# %%
# ------------------------------------------------------------------ 
# Plotting help from Amanda - don't run this
#create dfs for all runs which have the average diff in WTD across the run and the x/y loc of the well
minimum = 0
maximum = 50
#https://www.rapidtables.com/convert/color/rgb-to-hex.html
#GREYS ["#D2D2D2","#646464"]
#BLUES ["#AFDAFF","#3251A1"]
norm=plt.Normalize(-100,100)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#AFDAFF","#3251A1"])
norm2=plt.Normalize(-25,25)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#66C2A5","white","#FC8D62"])
plt.scatter(x = all_yr_avg['x_locs'] , y = all_yr_avg['y_locs'], s = 10.0, linewidth=.2,  c = all_yr_avg['diff'], edgecolor = "black",cmap = cmap2, norm=norm2) #well locations #x,y for plotting
plt.clim(-25, 25)#setting limits for color bar
plt.colorbar(label='Anomaly (m)', orientation="vertical", shrink=0.75)
plt.imshow(wtdflip[0,:,:],vmin=minimum, vmax=maximum,cmap = cmap, norm=norm)
plt.colorbar(label='Baseline Water Table Depth (m)', shrink=0.75)
#title = f'Baseline Model Performance Compared to \n44 Observation Wells'
#plt.title(title)
plt.xlabel('')
plt.ylabel('')
plt.xticks([]),plt.yticks([])
fig_name = f'./spatial_outputs/spatial_wtd_diff_all_yr_avg.png'
plt.savefig(fig_name, dpi=400, bbox_inches='tight')
plt.show()
plt.close()
# %%
