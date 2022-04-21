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
from optparse import Values
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
c_1 = '#8d5a99' # Reservation
c_2 = "#d7191c" # Regulated with CAP (Water Category Color)
c_3 = '#e77a47' # Regulated without CAP (Water Category Color)
c_4 = '#2cbe21' # Lower CO River - SW (Water Category Color)
c_5 = '#2f8c73' # Upper CO River - Mixed (Water Category Color)
c_6 = '#6db7e8' # SE - GW
c_7 = '#165782' # NW - GW (Water Category color)
c_8 = '#229ce8' # SC - GW
c_9 = '#1f78b4' # NE - GW
c_10 = '#41bf9e' # N - Mixed
c_11 = '#7adec4' # C - Mixed
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


# %% Plot in a four panel 1 column graph
ds = cat_wl2
minyear=2002
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 350
fsize = 14
ylabel = "Water Level (ft)"
linewidth = 2

# del ds.at[2015, 10]
ds.at[2015, 10] = None

# For the actual figure
fig, ax = plt.subplots(4,1,figsize=(12,15))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.05)
fig.supxlabel("Year", fontsize = 14, y=0.08)
# Panel numbers for easy adjusting
p1 = 0 # Panel 1
p2 = 1 # Panel 2
p3 = 2 # Panel 3
p4 = 3 # Panel 4
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[p1].plot(ds[2], label='Regulated with CAP', color=c_2, lw=linewidth) 
ax[p3].plot(ds[3], label='Regulated without CAP', color=c_3, lw=linewidth) 
ax[p1].plot(ds[4], color=c_4, label='Lower Colorado River - SW Dominated', lw=linewidth)
ax[p2].plot(ds[5], color=c_5, label='Upper Colorado River - Mixed', lw=linewidth)
#ax[p2].plot(ds[10], color=c_10, label='North - Mixed', lw=linewidth)
ax[p2].plot(ds[11], color=c_11, label='Central - Mixed', lw=3)
ax[p4].plot(ds[7], color=c_7, label='Northwest - GW Dominated', lw=linewidth)
ax[p3].plot(ds[9], color=c_9, label='Northeast - GW Dominated', lw=linewidth)
ax[p4].plot(ds[8], color=c_8, label='South central - GW Dominated', lw=linewidth)
ax[p3].plot(ds[6], color=c_6, label='Southeast - GW Dominated', lw=3)

ax[p1].set_xlim(minyear,maxyear)
ax[p2].set_xlim(minyear,maxyear)
ax[p3].set_xlim(minyear,maxyear)
ax[p4].set_xlim(minyear,maxyear)
ax[p1].set_ylim(max_y,min_y)
ax[p2].set_ylim(max_y,min_y)
ax[p3].set_ylim(max_y,min_y)
ax[p4].set_ylim(max_y,min_y)
ax[p1].grid(True)
ax[p2].grid(True)
ax[p3].grid(True)
ax[p4].grid(True)
#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)

# Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999

ax[p1].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[p1].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[p1].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax[p2].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[p2].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[p2].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax[p3].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[p3].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[p3].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax[p4].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax[p4].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax[p4].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)

# Wet years (2005 and 2010)
g = 2005
h = 2010
ax[p1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[p1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax[p2].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[p2].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax[p3].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[p3].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax[p4].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax[p4].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

ax[p1].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p2].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p3].legend(loc = [1.02, 0.30], fontsize = fsize)
ax[p4].legend(loc = [1.02, 0.20], fontsize = fsize)

# plt.savefig(outputpath+name+'_3panel', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
# plt.savefig(outputpath+name+'_4panel', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
plt.savefig(outputpath+name+'_4panel_drought', bbox_inches = 'tight') # bbox_inches makes sure the legend saves

# plt.savefig(outputpath+name+'_3panel_drought')

# %% Plot in a three panel graph, 1 column
ds = cat_wl2
minyear=1971
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 350
fsize = 14
ylabel = "Water Level (ft)"
linewidth = 2

# del ds.at[2015, 10]
ds.at[2015, 10] = None

# For the actual figure
fig, ax = plt.subplots(3,1,figsize=(12,15))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.05)
fig.supxlabel("Year", fontsize = 14, y=0.08)
# Panel numbers for easy adjusting
p1 = 0 # Panel 1
p2 = 1 # Panel 2
p3 = 2 # Panel 3
p4 = 3 # Panel 4
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[p1].plot(ds[2], label='Regulated with CAP', color=c_2, lw=linewidth) 
ax[p3].plot(ds[3], label='Regulated without CAP', color=c_3, lw=linewidth, zorder = 10) 
ax[p1].plot(ds[4], color=c_4, label='Lower Colorado River - SW Dominated', lw=linewidth)
ax[p2].plot(ds[5], color=c_5, label='Upper Colorado River - Mixed', lw=linewidth)
#ax[p2].plot(ds[10], color=c_10, label='North - Mixed', lw=linewidth)
ax[p2].plot(ds[11], color=c_11, label='Central - Mixed', lw=linewidth)
ax[p3].plot(ds[7], color=c_7, label='Northwest - GW Dominated', lw=linewidth)
# ax[p3].plot(ds[9], color=c_9, label='Northeast - GW Dominated', lw=linewidth)
# ax[p3].plot(ds[8], color=c_8, label='South central - GW Dominated', lw=linewidth)
ax[p3].plot(ds[6], color=c_6, label='Southeast - GW Dominated', lw=linewidth)

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

# ax[p1].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[p1].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[p1].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# ax[p2].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[p2].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[p2].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# ax[p3].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[p3].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[p3].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# # ax[p4].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# # ax[p4].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# # ax[p4].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)

# # Wet years (2005 and 2010)
# g = 2005
# h = 2010
# ax[p1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[p2].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p2].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[p3].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p3].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# # ax[p4].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# # ax[p4].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

ax[p1].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p2].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p3].legend(loc = [1.02, 0.30], fontsize = fsize)
# ax[p4].legend(loc = [1.02, 0.20], fontsize = fsize)

plt.savefig(outputpath+name+'_3panel_custom', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
# plt.savefig(outputpath+name+'_4panel', bbox_inches = 'tight') # bbox_inches makes sure the legend saves

# plt.savefig(outputpath+name+'_3panel_drought')
# %% --- Now making graphs for other things
# - Well Density (cumulative number of wells) over time
# - max screen depth over time (Casing_DEP vs Installed)
# - Number of new wells installed over time

# Re-read in after proper install date formatting, also had to fix NaN values in Regulation column
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

# %% Checking information about static_geo2

static_na = static_geo2[static_geo2['Regulation'].isna()]
print(static_na)

# %%
static_na['GEO_Region'].unique()

# %% 
Well_Depth = static_geo2[['WELL_DEPTH', 'INSTALLED', 'Combo_ID', 'In_year','GEOREGI_NU', 'Regulation', 'Water_CAT']]
#%%
Well_Depth

#%%
# Well_Depth.to_csv('../MergedData/Output_files/Final_WellDepth.csv')
#%%
# Well_Depth = pd.pivot_table(static_geo2, index=["In_year"], columns=["GEOREGI_NU"], values=["WELL_DEPTH"], dropna=False, aggfunc=np.mean)
Well_Depth = pd.pivot_table(static_geo2, index=["In_year"], columns=["Regulation"], values=["WELL_DEPTH"], dropna=False, aggfunc=np.mean)
state_depth = pd.pivot_table(static_geo2, index=["In_year"], values=["WELL_DEPTH"], dropna=False, aggfunc=np.mean)
Well_Depth.describe()

# %% Set shallow and drilling depths
shallow = 200
deep = 500

# %%
wd1 = static_geo2[(static_geo2["WELL_DEPTH"] > deep)]
wd2 = static_geo2[(static_geo2["WELL_DEPTH"] <= deep) & (static_geo2["WELL_DEPTH"] >= shallow)]
wd3 = static_geo2[(static_geo2["WELL_DEPTH"] < shallow)]

# st_wd1 = state_depth[(state_depth["WELL_DEPTH"] > deep)]
# st_wd2 = state_depth[(state_depth["WELL_DEPTH"] <= deep) & (static_geo2["WELL_DEPTH"] >= shallow)]
# st_wd3 = state_depth[(state_depth["WELL_DEPTH"] < shallow)]


# %%
wd1 = wd1.sort_values(by=['GEOREGI_NU'])
wd2 = wd2.sort_values(by=['GEOREGI_NU'])
wd3 = wd3.sort_values(by=['GEOREGI_NU'])
#%%
# wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["GEOREGI_NU"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
# wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["GEOREGI_NU"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
# wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["GEOREGI_NU"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

# wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
# wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
# wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["Regulation"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["Regulation"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["Regulation"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

# wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["Water_CAT"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
# wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["Water_CAT"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
# wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["Water_CAT"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

# %%
st_wdc1 = pd.pivot_table(wd1, index=["In_year"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
st_wdc2 = pd.pivot_table(wd2, index=["In_year"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
st_wdc3 = pd.pivot_table(wd3, index=["In_year"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
st_wdc1

#%% Exporting the Depth categories
# wdc1.to_csv('../MergedData/Output_files/Final_Welldepth' + str(deep) + 'plus.csv')
# wdc2.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'to' + str(deep) + '.csv')
# wdc3.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'minus.csv')

# %% Plotting fun
wdc1.plot()
wdc2.plot()
wdc3.plot()
# %% Plots for Water Source Lumping
ds = wdc3
name = 'Number of Shallow Wells (less than '+ str(shallow) +' ft)'
ylabel = "Well Count (#)"
minyear=1980
maxyear=2019
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax.plot(ds[labels[0]], label='CAP', color=c_2)
ax.plot(ds[labels[3]], label='Non-CAP', color=c_3)
ax.plot(ds[labels[1]], label='Groundwater ', color=c_7)
ax.plot(ds[labels[4]], label='Surface Water', color=c_4)
ax.plot(ds[labels[2]], label='Mixed GW/SW', color=c_5)

# ax.plot(st_wdc1, label='State Average', color='black')
ax.set_xlim(minyear,maxyear)
ax.grid(True)
ax.legend(fontsize = fsize)

plt.savefig(outputpath+name+'_WaterSource')
# %% Plots for Water Source Lumping
ds = wdc2
name = 'Number of Mid-range Wells (between '+ str(shallow) +' and '+ str(deep) +' ft)'
ylabel = "Well Count (#)"
minyear=1980
maxyear=2019
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax.plot(ds[labels[0]], label='CAP', color=c_2)
ax.plot(ds[labels[3]], label='Non-CAP', color=c_3)
ax.plot(ds[labels[1]], label='Groundwater ', color=c_7)
ax.plot(ds[labels[4]], label='Surface Water', color=c_4)
ax.plot(ds[labels[2]], label='Mixed GW/SW', color=c_5)

# ax.plot(st_wdc1, label='State Average', color='black')
ax.set_xlim(minyear,maxyear)
ax.grid(True)
ax.legend(fontsize = fsize)

plt.savefig(outputpath+name+'_WaterSource')
# %% Plots for Water Source Lumping
ds = wdc1
name = 'Number of Deep Wells (greater than '+ str(deep) +' ft)'
ylabel = "Well Count (#)"
minyear=1980
maxyear=2019
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax.plot(ds[labels[0]], label='CAP', color=c_2)
ax.plot(ds[labels[3]], label='Non-CAP', color=c_3)
ax.plot(ds[labels[1]], label='Groundwater ', color=c_7)
ax.plot(ds[labels[4]], label='Surface Water', color=c_4)
ax.plot(ds[labels[2]], label='Mixed GW/SW', color=c_5)

# ax.plot(st_wdc1, label='State Average', color='black')
ax.set_xlim(minyear,maxyear)
ax.grid(True)
ax.legend(fontsize = fsize)

plt.savefig(outputpath+name+'_WaterSource')

# %% Plots for Regulated vs. Unregulated Lumping
ds = wdc1
name = 'Number of Deep Wells (greater than '+ str(deep) +' ft)'
ylabel = "Well Count (#)"
minyear=1980
maxyear=2019
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax.plot(ds[labels[0]], label='Regulated', color=c_2)
ax.plot(ds[labels[1]], label='Unregulated', color=c_7)
# ax.plot(st_wdc1, label='State Average', color='black')
ax.set_xlim(minyear,maxyear)
ax.grid(True)
ax.legend(fontsize = fsize)

# plt.savefig(outputpath+name+'_regulation')

# %%
ds = wdc2
name = 'Number of Mid-range Wells (between '+ str(shallow) +' and '+ str(deep) +' ft)'
ylabel = "Well Count (#)"
minyear=1980
maxyear=2019
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax.plot(ds[labels[0]], label='Regulated', color=c_2)
ax.plot(ds[labels[1]], label='Unregulated', color=c_7)
ax.set_xlim(minyear,maxyear)
ax.grid(True)
ax.legend()

plt.savefig(outputpath+name+'_regulation')

# %%
ds = wdc3
name = 'Number of Shallow Wells (less than '+ str(shallow) +' ft)'
ylabel = "Well Count (#)"
minyear=1980
maxyear=2019
#min_y = -15
#max_y = 7
fsize = 14

columns = ds.columns
labels = ds.columns.tolist()
print(labels)

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax.plot(ds[labels[0]], label='Regulated', color=c_2)
ax.plot(ds[labels[1]], label='Unregulated', color=c_7)
ax.set_xlim(minyear,maxyear)
ax.grid(True)
ax.legend()

plt.savefig(outputpath+name+'_regulation')
# %%
# Plot all of the different depths 3 in a line
ds1 = wdc1
ds2 = wdc2
ds3 = wdc3

name = 'Well Depths over Time'
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
fig, ax = plt.subplots(1,3,figsize=(18,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax[0].plot(ds3[labels[0]], label='Regulated', color=c_2)
ax[0].plot(ds3[labels[1]], label='Unregulated', color=c_7)
ax[1].plot(ds2[labels[0]], label='Regulated', color=c_2)
ax[1].plot(ds2[labels[1]], label='Unregulated', color=c_7)
ax[2].plot(ds1[labels[0]], label='Regulated', color=c_2)
ax[2].plot(ds1[labels[1]], label='Unregulated', color=c_7)

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
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)

# plt.savefig(outputpath+name+'_4panel')
# %%
# Plot all of the deep wells in 4 panel
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
fig, ax = plt.subplots(4,figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
ax[0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[1].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[3].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[3].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
# ax[1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[3].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[1].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[1].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[2].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[2].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[0].set_xlim(minyear,maxyear)
ax[1].set_xlim(minyear,maxyear)
ax[2].set_xlim(minyear,maxyear)
ax[3].set_xlim(minyear,maxyear)

#ax[0].set_ylim(min_y,max_y)
#ax[1].set_ylim(min_y,max_y)
#ax[2].set_ylim(min_y,max_y)

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[3].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[0].legend(loc = [1.05, 0.40], fontsize = fsize)
ax[1].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[2].legend(loc = [1.05, 0.3], fontsize = fsize)
ax[3].legend(loc = [1.05, 0.3], fontsize = fsize)

# plt.savefig(outputpath+name+'_4panel')

# %%  Plot all of the deep wells in 2x2 panel
ds = wdc1
name = 'Number of Deep Wells (greater than '+ str(deep) +' ft)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
#min_y = -15
#max_y = 7
fsize = 10

columns = ds.columns
labels = ds.columns.tolist()
print(labels)
p0 = 0,0
p1 = 0,1
p2 = 1,0
p3 = 1,1

# For the actual figure
fig, ax = plt.subplots(2, 2, figsize=(18,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.09)
ax[p0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[p1].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[p2].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[p2].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
# ax[p1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[p2].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[p1].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[p1].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[p3].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[p3].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[p0].set_xlim(minyear,maxyear)
ax[p1].set_xlim(minyear,maxyear)
ax[p2].set_xlim(minyear,maxyear)
ax[p3].set_xlim(minyear,maxyear)

#ax[p0].set_ylim(min_y,max_y)
#ax[p1].set_ylim(min_y,max_y)
#ax[p2].set_ylim(min_y,max_y)

ax[p0].grid(True)
ax[p1].grid(True)
ax[p2].grid(True)
ax[p3].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[p0].legend(loc = [0.02, 0.8], fontsize = fsize)
ax[p1].legend(loc = [0.02, 0.73], fontsize = fsize)
ax[p2].legend(loc = [0.5, 0.73], fontsize = fsize)
ax[p3].legend(loc = [0.02, 0.8], fontsize = fsize)

plt.savefig(outputpath+name+'_2by2panel')
# %% Mid-range wells 3 panel
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

plt.savefig(outputpath+name+'_3panel')

# %%  Plot all of the midrange wells in 2x2 panel
ds = wdc2
name = 'Number of Mid-range Wells (between '+ str(shallow) +' and '+ str(deep)+' ft.)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
#min_y = -15
#max_y = 7
fsize = 10

columns = ds.columns
labels = ds.columns.tolist()
print(labels)
p0 = 0,0
p1 = 0,1
p2 = 1,0
p3 = 1,1

# For the actual figure
fig, ax = plt.subplots(2, 2, figsize=(18,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.09)
ax[p0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[p1].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[p0].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[p2].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
# ax[p1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[p2].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[p1].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[p1].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[p3].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[p3].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[p0].set_xlim(minyear,maxyear)
ax[p1].set_xlim(minyear,maxyear)
ax[p2].set_xlim(minyear,maxyear)
ax[p3].set_xlim(minyear,maxyear)

#ax[p0].set_ylim(min_y,max_y)
#ax[p1].set_ylim(min_y,max_y)
#ax[p2].set_ylim(min_y,max_y)

ax[p0].grid(True)
ax[p1].grid(True)
ax[p2].grid(True)
ax[p3].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[p0].legend(loc = [0.02, 0.8], fontsize = fsize)
ax[p1].legend(loc = [0.02, 0.73], fontsize = fsize)
ax[p2].legend(loc = [0.6, 0.8], fontsize = fsize)
ax[p3].legend(loc = [0.02, 0.8], fontsize = fsize)

plt.savefig(outputpath+name+'_2by2panel')

# %% Shallow wells in 4 panel (1 column)
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

plt.savefig(outputpath+name+'_3panel')

# %%  Plot all of the shallow wells in 2x2 panel
ds = wdc3
name = 'Number of Shallow Wells (less than '+ str(shallow) +' ft)'
ylabel = "Well Count (#)"
minyear=1975
maxyear=2020
#min_y = -15
#max_y = 7
fsize = 10

columns = ds.columns
labels = ds.columns.tolist()
print(labels)
p0 = 0,0
p1 = 0,1
p2 = 1,0
p3 = 1,1

# For the actual figure
fig, ax = plt.subplots(2, 2, figsize=(18,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.09)
ax[p0].plot(ds[labels[1]], label='Regulated with CAP', color=c_2) 
ax[p1].plot(ds[labels[2]], label='Regulated without CAP', color=c_3) 
ax[p0].plot(ds[labels[3]], color=c_4, label='Lower Colorado River - SW Dominated')
ax[p2].plot(ds[labels[4]], color=c_5, label='Upper Colorado River - Mixed')
# ax[p1].plot(ds[labels[9]], color=c_10, label='North - Mixed')
ax[p2].plot(ds[labels[10]], color=c_11, label='Central - Mixed')
ax[p1].plot(ds[labels[6]], color=c_7, label='Northwest - GW Dominated')
ax[p1].plot(ds[labels[8]], color=c_9, label='Northeast - GW Dominated')
ax[p3].plot(ds[labels[7]], color=c_8, label='South central - GW Dominated')
ax[p3].plot(ds[labels[5]], color=c_6, label='Southeast - GW Dominated')

ax[p0].set_xlim(minyear,maxyear)
ax[p1].set_xlim(minyear,maxyear)
ax[p2].set_xlim(minyear,maxyear)
ax[p3].set_xlim(minyear,maxyear)

#ax[p0].set_ylim(min_y,max_y)
#ax[p1].set_ylim(min_y,max_y)
#ax[p2].set_ylim(min_y,max_y)

ax[p0].grid(True)
ax[p1].grid(True)
ax[p2].grid(True)
ax[p3].grid(True)

#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)
ax[p0].legend(loc = [0.02, 0.8], fontsize = fsize)
ax[p1].legend(loc = [0.6, 0.75], fontsize = fsize)
ax[p2].legend(loc = [0.58, 0.8], fontsize = fsize)
ax[p3].legend(loc = [0.02, 0.8], fontsize = fsize)

plt.savefig(outputpath+name+'_2by2panel')


# %%
new_wells = pd.pivot_table(static_geo2, index=["In_year"], columns=["GEOREGI_NU"], values=["INSTALLED"], dropna=False, aggfunc=len)
new_wells_reg = pd.pivot_table(static_geo2, index=["In_year"], columns=["Regulation"], values=["INSTALLED"], dropna=False, aggfunc=len)
new_wells_watercat = pd.pivot_table(static_geo2, index=["In_year"], columns=["Water_CAT"], values=["INSTALLED"], dropna=False, aggfunc=len)
new_wells_reg
new_wells_watercat
# %%
# new_wells.to_csv('../MergedData/Output_files/Final_NewWells.csv')
new_wells_reg.to_csv('../MergedData/Output_files/Final_NewWells_regulation.csv')
new_wells_watercat.to_csv('../MergedData/Output_files/Final_NewWells_watercat.csv')

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
new_wells2
new_wells2 = new_wells2.reset_index()
new_wells2 = new_wells2.iloc[1:, :]
#new_wells2 = new_wells2.rename(columns = {'GEOREGI_NU':'Year'})
new_wells2 = new_wells2.set_index('GEOREGI_NU')
new_wells2

# %%
new_wells_reg2 = pd.read_csv('../MergedData/Output_files/Final_NewWells_regulation.csv',
                        header=2,
                        names = ['R','U']
                        , index_col=0)
# new_wells_reg2 = new_wells_reg2.reset_index()
new_wells_reg2 = new_wells_reg2.iloc[1:, :]
# new_wells_reg2 = new_wells_reg2.set_index('Year')
new_wells_reg2

# %%
new_wells_watercat2 = pd.read_csv('../MergedData/Output_files/Final_NewWells_watercat.csv',
                        header=1,
                        names = ['CAP','GW','Mix','No_CAP','SW']
                        , index_col=0)
# new_wells_watercat2 = new_wells_watercat2.reset_index()
new_wells_watercat2 = new_wells_watercat2.iloc[1:, :]
# new_wells_watercat2 = new_wells_watercat2.set_index('Year')
new_wells_watercat2

# %%

# %% Calculate the region area
# to double check the coordinate system is in meters, georeg.crs
# 
# https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas
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

# %% Area for other categories
georeg_area_reg = pd.pivot_table(georeg, columns=["Regulation"], values=["area"], dropna=False, aggfunc=np.sum)
del georeg_area_reg['NA']
georeg_area_reg

# %%
georeg_area_watercat = pd.pivot_table(georeg, columns=["Water_CAT"], values=["area"], dropna=False, aggfunc=np.sum)
del georeg_area_watercat['NA']
georeg_area_watercat
# %% Densities for new wells
# df1/df2.values[0,:]
well_densities = new_wells2/georeg_area.values[0,:]
well_densities

# %% Densities for regulated regions
well_densities_reg = new_wells_reg2/georeg_area_reg.values[0,:]
well_densities_reg

# %% Densities for SW
well_densities_watercat = new_wells_watercat2/georeg_area_watercat.values[0,:]
well_densities_watercat

# %% By depth for regulated or water category, depending on what I turned on above
dens_wdc1= wdc1/georeg_area_reg.values[0,:]
dens_wdc2= wdc2/georeg_area_reg.values[0,:]
dens_wdc3= wdc3/georeg_area_reg.values[0,:]

# dens_wdc1= wdc1/georeg_area_watercat.values[0,:]
# dens_wdc2= wdc2/georeg_area_watercat.values[0,:]
# dens_wdc3= wdc3/georeg_area_watercat.values[0,:]

dens_wdc1
print(dens_wdc1.sum())

# %% ------------------------------------------------------------------------
# This whole thing was to double check if my numbers were correct, skip this.
well_densities_reg_total = well_densities_reg.sum()
well_densities_reg_total

# %%
# new_wells_reg_total = new_wells_reg2.sum()
# new_wells_reg_total = pd.pivot_table(new_wells_reg2, index = dropna=False, aggfunc=np.sum)
new_wells_reg_total = pd.DataFrame()
new_wells_reg_total['R'] = new_wells_reg2['R'].sum()
new_wells_reg_total['U'] = new_wells_reg2['U'].sum()
new_wells_reg_total

# %% df.loc['Column_Total']= df.sum(numeric_only=True, axis=0)
new_wells_reg_total = new_wells_reg2
new_wells_reg_total.loc['Column_Total']= new_wells_reg_total.sum(numeric_only=True, axis=0)
# new_wells_reg_total = new_wells_reg_total.loc['Column_Total']
new_wells_reg_total

# %%
test = pd.DataFrame(new_wells_reg2.sum(numeric_only=True, axis=0))
test = test.transpose()
test
# %%
test_welldensit_reg = test/georeg_area_reg.values[0,:]
test_welldensit_reg
# --------------------------------------------------------------------------

# %%
well_densities['3'].plot()

# %%
well_densities1 = well_densities.reset_index()
well_densities1['GEOREGI_NU'] = pd.to_numeric(well_densities1['GEOREGI_NU'])
well_densities1['GEOREGI_NU'] = well_densities1['GEOREGI_NU'].astype(int)
well_densities1.set_index('GEOREGI_NU', inplace=True)
well_densities1.info()

# %%
ds = well_densities1
name = 'Well Densities Per Region (#/km2)'
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

# %% Trying to different types of charts
# Labels for reference and different color orders
georeg = georeg.sort_values(by=['GEOREGI_NU'])
labels = dict(zip(georeg.GEOREGI_NU, georeg.GEO_Region))
# del labels[1]
print(labels)
#%%
barcolors = [c_2, c_3, c_4, c_5, c_10, c_11, c_7, c_9, c_8, c_6]
barhcolors = [c_6, c_8, c_9, c_7, c_3, c_11, c_10, c_5, c_4, c_2]
bar_regc = [c_2,c_7]
bar_watercatc = [c_2,c_3,c_4,c_5,c_7]
piecolors_unreg = [c_6, c_8, c_9, c_7, c_11, c_10, c_5, c_4]
# %% Normal Bar Chart
ds = well_densities1
ds1 = pd.DataFrame()
ds1['Reg. with CAP'] = ds['2']
ds1['Reg. without CAP'] = ds['3']
ds1['Lower CO River'] = ds['4']
ds1['Upper CO River'] = ds['5']
ds1['North'] = ds['10']
ds1['Central'] = ds['11']
ds1['Northwest'] = ds['7']
ds1['Northeast'] = ds['9']
ds1['South-central'] = ds['8']
ds1['Southeast'] = ds['6']

plt.figure(figsize = (10,6))
plt.title('Well Densities Per Region')
plt.bar(ds1.columns, ds1.sum(), color = barcolors, zorder=4)
plt.ylabel('Number of Wells / km^2')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

# %% Horizontal Bar Chart
ds = well_densities1
ds1 = pd.DataFrame()
ds1['Southeast'] = ds['6']
ds1['South-central'] = ds['8']
ds1['Northeast'] = ds['9']
ds1['Northwest'] = ds['7']
ds1['Reg. without CAP'] = ds['3']
ds1['Central'] = ds['11']
ds1['North'] = ds['10']
ds1['Upper Colorado River'] = ds['5']
ds1['Lower Colorado River'] = ds['4']
ds1['Regulated with CAP'] = ds['2']

plt.figure(figsize = (10,6))
plt.title('Well Densities Per Region')
plt.barh(ds1.columns, ds1.sum(), color = barhcolors, zorder=4)
plt.xlabel('Number of Wells / km^2')
plt.xticks()
plt.grid(axis='x', linewidth=0.5, zorder=0)

# %% Normal Bar Chart for Deep wells with regulation lumping
ds = wdc1
columns = ds.columns
labels = ds.columns.tolist()

print(labels)

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

plt.figure(figsize = (6,6))
plt.title('Deep Wells (Greater than 500 ft)')
plt.bar(ds1.columns, ds1.sum(), 
        color = bar_regc, 
        zorder=4)
plt.ylabel('Number of Wells (#)')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

# %%
ds = wdc2
columns = ds.columns
labels = ds.columns.tolist()

print(labels)

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

plt.figure(figsize = (6,6))
plt.title('Midrange Wells (Between 200 and 500 ft)')
plt.bar(ds1.columns, ds1.sum(), 
        color = bar_regc, 
        zorder=4)
plt.ylabel('Number of Wells (#)')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

# %%
ds = wdc3
columns = ds.columns
labels = ds.columns.tolist()

print(labels)

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

plt.figure(figsize = (6,6))
plt.title('Shallow Wells (Less than 200 ft)')
plt.bar(ds1.columns, ds1.sum(), 
        color = bar_regc, 
        zorder=4)
plt.ylabel('Number of Wells (#)')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

# %% -- Grouped bar chart - Had to create some summarazing dataframes --
# Check the commented code to turn on whichever graph you want to make
#   - dens = well densities
#   - wdc1 = water depth category 1 (deep)
#     wdc2 = midrange
#     wdc3 = shallow

# Below is for Groundwater Regulation
ds = wdc1.copy()
# ds = dens_wdc1.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

dft1 = ds1.copy()
dft1


ds = wdc2.copy()
# ds = dens_wdc2.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

dft2 = ds1.copy()
dft2


ds = wdc3.copy()
# ds = dens_wdc3.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

dft3 = ds1.copy()
dft3

df1 = pd.DataFrame(dft1.sum())
df1 = df1.transpose()
df1 = df1.reset_index()
df1['index'] = 'Deep'
df1.set_index('index', inplace=True)
df1

df2 = pd.DataFrame(dft2.sum())
df2 = df2.transpose()
df2 = df2.reset_index()
df2['index'] = 'Midrange'
df2.set_index('index', inplace=True)
df2

df3 = pd.DataFrame(dft3.sum())
df3 = df3.transpose()
df3 = df3.reset_index()
df3['index'] = 'Shallow'
df3.set_index('index', inplace=True)
df3

df_test = df3.append([df2,df1])
df_test = df_test.transpose()
df_test = df_test.rename_axis(None,axis=1)
df_test

# group_colors = ['cornflowerblue','slategrey','darkblue']
group_colors = ['lightsteelblue','cornflowerblue','darkblue']

# name = 'Well Densities by Groundwater Regulation'
# horlabel = 'Well Densities (well/km^2)'
name = 'Number of wells by Groundwater Regulation'
horlabel = 'Number of Wells (#)'
fsize = 14

df_test.plot(figsize = (12,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(fontsize = fsize)

plt.savefig(outputpath+name+'groupedchart')


# %% Below is for Water Access Category
# ds = wdc1.copy()
ds = dens_wdc1.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW Regulated'] = ds[labels[3]]
ds1['GW'] = ds[labels[1]]

dft1 = ds1.copy()
dft1


# ds = wdc2.copy()
ds = dens_wdc2.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW Regulated'] = ds[labels[3]]
ds1['GW'] = ds[labels[1]]

dft2 = ds1.copy()
dft2


# ds = wdc3.copy()
ds = dens_wdc3.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW Regulated'] = ds[labels[3]]
ds1['GW'] = ds[labels[1]]

dft3 = ds1.copy()
dft3

df1 = pd.DataFrame(dft1.sum())
df1 = df1.transpose()
df1 = df1.reset_index()
df1['index'] = 'Deep'
df1.set_index('index', inplace=True)
df1

df2 = pd.DataFrame(dft2.sum())
df2 = df2.transpose()
df2 = df2.reset_index()
df2['index'] = 'Midrange'
df2.set_index('index', inplace=True)
df2

df3 = pd.DataFrame(dft3.sum())
df3 = df3.transpose()
df3 = df3.reset_index()
df3['index'] = 'Shallow'
df3.set_index('index', inplace=True)
df3

df_test = df3.append([df2,df1])
df_test = df_test.transpose()
df_test = df_test.rename_axis(None,axis=1)
df_test

# group_colors = ['cornflowerblue','slategrey','darkblue']
group_colors = ['lightsteelblue','cornflowerblue','darkblue']

# name = 'Well Densities by Access to Surface Water'
horlabel = 'Well Densities (well/km^2)'
# name = 'Number of wells by Access to Surface Water'
# horlabel = 'Number of Wells (#)'
fsize = 14

df_test.plot(figsize = (12,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(fontsize = fsize)

# plt.savefig(outputpath+name+'groupedchart')


# %% Normal Bar Chart for Deep wells with Water Category lumping
ds = wdc3
columns = ds.columns
labels = ds.columns.tolist()
name = 'Shallow Wells (less than 200ft)'

print(labels)

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['No_CAP'] = ds[labels[3]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW'] = ds[labels[1]]

plt.figure(figsize = (6,6))
plt.title(name)
plt.bar(ds1.columns, ds1.sum(), 
        color = bar_watercatc, 
        zorder=4)
plt.ylabel('Number of Wells (#)')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

plt.savefig(outputpath+name+'barchart_watercat')

# %%
ds = wdc2
columns = ds.columns
labels = ds.columns.tolist()
name = 'Midrange Wells (Between 200 and 500 ft)'

print(labels)

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['No_CAP'] = ds[labels[3]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW'] = ds[labels[1]]

plt.figure(figsize = (6,6))
plt.title(name)
plt.bar(ds1.columns, ds1.sum(), 
        color = bar_watercatc, 
        zorder=4)
plt.ylabel('Number of Wells (#)')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

plt.savefig(outputpath+name+'barchart_watercat')

# %%
ds = wdc1
columns = ds.columns
labels = ds.columns.tolist()
name = 'Deep Wells (Greater than 500 ft)'

print(labels)

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['No_CAP'] = ds[labels[3]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW'] = ds[labels[1]]

plt.figure(figsize = (6,6))
plt.title(name)
plt.bar(ds1.columns, ds1.sum(), 
        color = bar_watercatc, 
        zorder=4)
plt.ylabel('Number of Wells (#)')
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)

plt.savefig(outputpath+name+'barchart_watercat')

# %% Plotting a pie chart for Regulation
ds = well_densities_reg
name = 'Well Densities (well per square km)'
columns = ds.columns
labels = ds.columns.tolist()

print(labels)

ds1 = pd.DataFrame()
ds1['Regulated'] = ds[labels[0]]
ds1['Unregulated'] = ds[labels[1]]

ds2 = ds1.sum()
# values = ds2

# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#     return my_autopct

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = pct*total/100.0
        return '{v:.3f}'.format(p=pct,v=val)
    return my_autopct

ds2.plot(figsize = (7,7), kind='pie',subplots=True, colors=bar_regc, 
        title=name,
        fontsize=14
        ,autopct=make_autopct(ds2)
        # ,autopct='%.2f'
        # ,autopct=['1.00','0.25'] # this doesn't work
        )
plt.gca().set_ylabel('')
plt.savefig(outputpath+name+'pie_reg')

# %% Plotting a pie chart for Water Category
# ds = new_wells_watercat
ds = well_densities_watercat
# name = 'Number of wells'
name = 'Well Densities (well per square km)'
columns = ds.columns
labels = ds.columns.tolist()

print(labels)

ds1 = pd.DataFrame()
ds1['CAP'] = ds[labels[0]]
ds1['GW Regulated'] = ds[labels[3]]
ds1['SW'] = ds[labels[4]]
ds1['Mixed'] = ds[labels[2]]
ds1['GW Unregulated'] = ds[labels[1]]

ds2 = ds1.sum()
# values = ds2

# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#     return my_autopct

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = pct*total/100.0
        return '{v:.3f}'.format(p=pct,v=val)
    return my_autopct

ds2.plot(figsize = (7,7),kind='pie',subplots=True, colors=bar_watercatc, 
        # title=name,
        autopct=make_autopct(ds2))
        # autopct='%1.1f%%')
plt.gca().set_ylabel('')
plt.savefig(outputpath+name+'pie_watercat')

# %% Plotting a pie chart for georegions
ds = new_wells
name = 'Number of wells for Groundwater dominated Regions'
columns = ds.columns
labels = ds.columns.tolist()

print(labels)

ds1 = pd.DataFrame()
ds1['Southeast'] = ds[labels[5]]
ds1['South-central'] = ds[labels[7]]
ds1['Northeast'] = ds[labels[8]]
ds1['Northwest'] = ds[labels[6]]
# ds1['Reg. without CAP'] = ds[labels[2]]
ds1['Central'] = ds[labels[10]]
ds1['North'] = ds[labels[9]]
ds1['Upper Colorado River'] = ds[labels[4]]
ds1['Lower Colorado River'] = ds[labels[3]]
# ds1['Regulated with CAP'] = ds[labels[1]]

ds2 = ds1.sum()
# values = ds2

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

ds2.plot(figsize = (10,10),kind='pie',subplots=True, colors=piecolors_unreg, 
        # title=name,
        autopct=make_autopct(ds2))
        # autopct='%1.1f%%')
plt.gca().set_ylabel('')
plt.savefig(outputpath+name+'pie_unregulated')

# %% -- Linear regression --
# This is testing whether or not the slope is positive or negative (2-way)
#       For our purposes, time is the x variable and y is
#       1. Water Levels
#       2. Number of Wells
#       3. Well Depths

# Actual documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# Tutorial from https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/

# For Water Levels
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
                              'std_err':std_err,
                              'mean': np.mean(y),
                              'var': np.var(y)}, 
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
stats.index = col.values()
stats1 = stats.transpose()
del stats1['Reservation']
stats1

# %% 2. For New Wells
ds = new_wells_reg2
data_type = "New Wells"
betterlabels = ['Regulated','Unregulated'] 
min_yr = 1993
mx_yr = 2019
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

# -- For Multiple years --
# Name = "Linear Regression during Non-drought years for " + data_type
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# #f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in dryyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# print(f)
# ------------------------

stats = pd.DataFrame()
# for i in range(1, 12, 1):
for i in column_list:
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        # print('Georegion Number: ', i, '\n', 
        #        'slope = ', slope, '\n', 
        #        'intercept = ', intercept, '\n', 
        #        'r^2 = ', r_value, '\n', 
        #        'p-value = ', p_value, '\n', 
        #        'std error = ', std_err)
        
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)
        # xf = np.linspace(min(x),max(x),100)
        # xf1 = xf.copy()
        # xf1 = pd.to_datetime(xf1)
        # yf = (slope*xf)+intercept
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(xf1, yf,label='Linear fit', lw=3)
        # df.plot(ax=ax,marker='o', ls='')
        # ax.set_ylim(max(y),0)
        # ax.legend()

stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)

# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope','Regulated'], 2)
m2 = round(stats1.loc['slope','Unregulated'], 2)
yint1 = round(stats1.loc['int','Regulated'], 2)
yint2 = round(stats1.loc['int','Unregulated'], 2)
pval1 = round(stats1.loc['p_val', 'Regulated'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2

f, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
ds.plot(ax=ax,marker='o', ls='', label=betterlabels)
ax.set_xlim(min_yr, mx_yr)
ax.set_title(data_type)
plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/'+Name+'.csv')

# 3. Well Depths
# == For Shallow ==
ds = wdc3
data_type = "Shallow Wells"
betterlabels = ['Regulated','Unregulated'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
# for i in range(1, 12, 1):
for i in column_list:
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        # print('Georegion Number: ', i, '\n', 
        #        'slope = ', slope, '\n', 
        #        'intercept = ', intercept, '\n', 
        #        'r^2 = ', r_value, '\n', 
        #        'p-value = ', p_value, '\n', 
        #        'std error = ', std_err)
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)
        # xf = np.linspace(min(x),max(x),100)
        # xf1 = xf.copy()
        # xf1 = pd.to_datetime(xf1)
        # yf = (slope*xf)+intercept
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(xf1, yf,label='Linear fit', lw=3)
        # df.plot(ax=ax,marker='o', ls='')
        # ax.set_ylim(max(y),0)
        # ax.legend()

stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)

# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope','Regulated'], 2)
m2 = round(stats1.loc['slope','Unregulated'], 2)
yint1 = round(stats1.loc['int','Regulated'], 2)
yint2 = round(stats1.loc['int','Unregulated'], 2)
pval1 = round(stats1.loc['p_val', 'Regulated'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2

f, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
ds.plot(ax=ax,marker='o', ls='', label=betterlabels)
ax.set_xlim(min_yr, mx_yr)
ax.set_title(data_type)
plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/'+Name+'.csv')

# == For Midrange ==
ds = wdc2
data_type = "Midrange Wells"
betterlabels = ['Regulated','Unregulated'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
# for i in range(1, 12, 1):
for i in column_list:
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        # print('Georegion Number: ', i, '\n', 
        #        'slope = ', slope, '\n', 
        #        'intercept = ', intercept, '\n', 
        #        'r^2 = ', r_value, '\n', 
        #        'p-value = ', p_value, '\n', 
        #        'std error = ', std_err)
        
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)
        # xf = np.linspace(min(x),max(x),100)
        # xf1 = xf.copy()
        # xf1 = pd.to_datetime(xf1)
        # yf = (slope*xf)+intercept
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(xf1, yf,label='Linear fit', lw=3)
        # df.plot(ax=ax,marker='o', ls='')
        # ax.set_ylim(max(y),0)
        # ax.legend()

stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)

# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope','Regulated'], 2)
m2 = round(stats1.loc['slope','Unregulated'], 2)
yint1 = round(stats1.loc['int','Regulated'], 2)
yint2 = round(stats1.loc['int','Unregulated'], 2)
pval1 = round(stats1.loc['p_val', 'Regulated'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2

f, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
ds.plot(ax=ax,marker='o', ls='', label=betterlabels)
ax.set_xlim(min_yr, mx_yr)
ax.set_title(data_type)
plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/'+Name+'.csv')

# == For Deep wells ==
ds = wdc1
data_type = "Deep Wells"
betterlabels = ['Regulated','Unregulated'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
# for i in range(1, 12, 1):
for i in column_list:
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
        # print('Georegion Number: ', i, '\n', 
        #        'slope = ', slope, '\n', 
        #        'intercept = ', intercept, '\n', 
        #        'r^2 = ', r_value, '\n', 
        #        'p-value = ', p_value, '\n', 
        #        'std error = ', std_err)      
        stats = stats.append({'slope': slope, 
                              'int':intercept, 
                              'rsq':r_value*r_value, 
                              'p_val':p_value, 
                              'std_err':std_err, 
                              'mean': np.mean(y),
                              'var': np.var(y),
                              'sum': np.sum(y)
                              },
                              ignore_index=True)
        # xf = np.linspace(min(x),max(x),100)
        # xf1 = xf.copy()
        # xf1 = pd.to_datetime(xf1)
        # yf = (slope*xf)+intercept
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(xf1, yf,label='Linear fit', lw=3)
        # df.plot(ax=ax,marker='o', ls='')
        # ax.set_ylim(max(y),0)
        # ax.legend()


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)

# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
#xf1 = pd.to_datetime(xf1)
m1 = round(stats1.loc['slope','Regulated'], 2)
m2 = round(stats1.loc['slope','Unregulated'], 2)
yint1 = round(stats1.loc['int','Regulated'], 2)
yint2 = round(stats1.loc['int','Unregulated'], 2)
pval1 = round(stats1.loc['p_val', 'Regulated'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2

f, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
ds.plot(ax=ax,marker='o', ls='', label=betterlabels)
ax.set_xlim(min_yr, mx_yr)
ax.set_title(data_type)
plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/'+Name+'.csv')

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
