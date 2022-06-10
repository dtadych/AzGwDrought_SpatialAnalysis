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
from matplotlib.transforms import Bbox
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

# %% Exporting or reading in the static geodatabase instead of rerunning
# static_geo.to_csv('../MergedData/Output_files/Final_Static_geodatabase_allwells.csv')
# filename = "Final_Static_geodatabase_allwells.csv"
# filepath = os.path.join(outputpath, filename)
# static_geo = gp.read_file(filepath)
# static_geo

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
# cat_wl = combo.groupby(['GEO_Region', 'GEOREGI_NU']).mean()
# cat_wl = combo.groupby(['GEOREGI_NU']).mean()
cat_wl = combo.groupby(['Regulation']).mean()
# cat_wl = combo.groupby(['Water_CAT']).mean()

cat_wl

# %% 
cat_wl2 = cat_wl.copy()
cat_wl2

# %% Skip this if you're checking Depth to Water based on regulation
cat_wl2 = cat_wl.sort_values(by=['GEOREGI_NU'])
cat_wl2

# %% Clean up the dataframe for graphing
del cat_wl2['GEOREGI_NU']
cat_wl2 = cat_wl2[1:]
# del cat_wl2['GEO_Region']
cat_wl2
# %%
# cat_wl2 = cat_wl2.set_index("GEOREGI_NU")
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
cat_wl2.to_csv('../MergedData/Output_files/Waterlevels_Regulation.csv')

# %% Creating dictionary of labels
labels = cat_wl2.columns.tolist()
# georeg = georeg.sort_values(by=['GEOREGI_NU'])
# labels = dict(zip(georeg.GEOREGI_NU, georeg.GEO_Region))
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

#%% Plot by Groundwater Regulation (line 129)
ds = cat_wl2
minyear=1975
maxyear=2020
name = "Average Depth to Water from " + str(minyear) + " to " + str(maxyear) + ' by Groundwater Regulation'
min_y = 0
max_y = 300
fsize = 14

fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds['R'], label='GW Regulated', color=c_2) 
ax.plot(ds['U'], label='GW Unregulated', color=c_7) 
ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
ax.grid(True)
ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
# # Drought Year Shading
a = 1988.5
b = 1990.5
c = 1995.5
d = 1996.5
e = 2001.5
f = 2003.5
g = 2005.5
h = 2007.5
i = 2011.5
j = 2014.5
k = 2017.5
l= 2018.5
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_byregulation', bbox_inches='tight')
plt.savefig(outputpath+name+'_byregulation_Drought', bbox_inches='tight')


#%% Plot by access to surfacewater
ds = cat_wl2
minyear=1975
maxyear=2020
name = "Average Depth to Water from " + str(minyear) + " to " + str(maxyear) + " by Access to SW"
min_y = 0
max_y = 300
fsize = 14

fig, ax = plt.subplots(figsize = (16,9))
ax.plot(ds['CAP'], label='CAP', color=c_2)
ax.plot(ds['No_CAP'], label='Regulated GW', color=c_3) 
ax.plot(ds['GW'], label='Unregulated GW', color=c_7) 
ax.plot(ds['Mix'], label='Mixed SW/GW', color=c_5) 
ax.plot(ds['SW'], label='Surface Water', color=c_4) 

ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
# Drought Year Shading
a = 1975
b = 1977.5
c = 1980.5
d = 1981.5
e = 1988.5
f = 1990.5
g = 1995.5
h = 1997.5
i = 1998.5
j = 2004.5
k = 2005.5
l = 2009.5
m = 2010.5
n = 2018.5
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(m, n, color=drought_color, alpha=0.5, lw=0)

# # Wet years (2005 and 2010)
# g = 2005
# h = 2010
# ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax.minorticks_on()

fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_Drought', bbox_inches='tight')
# plt.savefig(outputpath+name+'_byGW', bbox_inches='tight')
# plt.savefig(outputpath+name+'_bySW', bbox_inches='tight')
# plt.savefig(outputpath+name+'_5', bbox_inches='tight')

#%% Plotting individual divisions
ds = cat_wl2
minyear=1975
maxyear=2020
name = "Average Depth to Water from " + str(minyear) + " to " + str(maxyear)
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
ax.set(title=name, xlabel='Year', ylabel='Depth to Water (ft)')
ax.legend(loc = [1.04, 0.40])

#%% Plot just the regulated
fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds[2.0], label='Regulated with CAP', color=c_2) 
ax.plot(ds[3.0], label='Regulated without CAP', color=c_3) 
ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
ax.grid(True)
ax.set(title=name, xlabel='Year', ylabel='Depth to Water (ft)')
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
ax.set(title=name, xlabel='Year', ylabel='Depth to Water (ft)')
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
ax.set(title=name, xlabel='Year', ylabel='Depth to Water (ft)')
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
ylabel = "Depth to Water (ft)"

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
minyear=1975
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 350
fsize = 14
ylabel = "Depth to Water (ft)"
linewidth = 2

# del ds.at[2015, 10]
# ds.at[2015, 10] = None

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
ax[p1].plot(ds[4], color=c_4, label='Lower Colorado River - SW Dominated', lw=linewidth)
ax[p2].plot(ds[5], color=c_5, label='Upper Colorado River - Mixed', lw=linewidth)
#ax[p2].plot(ds[10], color=c_10, label='North - Mixed', lw=linewidth)
ax[p2].plot(ds[11], color=c_11, label='Central - Mixed', lw=3)
ax[p4].plot(ds[7], color=c_7, label='Northwest - GW Dominated', lw=linewidth)
ax[p3].plot(ds[9], color=c_9, label='Northeast - GW Dominated', lw=linewidth)
ax[p4].plot(ds[8], color=c_8, label='South central - GW Dominated', lw=linewidth)
ax[p3].plot(ds[6], color=c_6, label='Southeast - GW Dominated', lw=3)
ax[p3].plot(ds[3], label='Regulated without CAP', color=c_3, lw=linewidth) 
ax[p4].plot(ds[3], label='Regulated without CAP', color=c_3, lw=linewidth) 


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
# ax[p4].axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# ax[p4].axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# ax[p4].axvspan(e, f, color=drought_color, alpha=0.5, lw=0)

# # Wet years (2005 and 2010)
# g = 2005
# h = 2010
# ax[p1].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p1].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[p2].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p2].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[p3].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p3].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
# ax[p4].axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax[p4].axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

ax[p1].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p2].legend(loc = [1.02, 0.40], fontsize = fsize)
ax[p3].legend(loc = [1.02, 0.30], fontsize = fsize)
ax[p4].legend(loc = [1.02, 0.20], fontsize = fsize)

# plt.savefig(outputpath+name+'_3panel', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
# plt.savefig(outputpath+name+'_4panel', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
plt.savefig(outputpath+name+'_4panel_1col', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_3panel_drought')
# %% Plot in a four panel 1 column graph
ds = cat_wl2
minyear=2002
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 350
fsize = 14
ylabel = "Depth to Water (ft)"
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
ylabel = "Depth to Water (ft)"
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

# %% -- Linear regression --
# This is testing whether or not the slope is positive or negative (2-way)
#       For our purposes, time is the x variable and y is
#       1. Depth to Water
#       2. Number of Wells
#       3. Well Depths

# Actual documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# Tutorial from https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/

# For Depth to Water of georegions
ds = cat_wl2
min_yr = 2002
mx_yr = 2020
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression:"
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]

# -- For Multiple years --
# Name = "Linear Regression for Non-drought years: "
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# #f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in dryyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# print(f)
# -----------------------

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
stats1 = stats.transpose()
stats1

# %%
# For Depth to Water by SW Access
ds = cat_wl2
data_type = "Depth to Water"
min_yr = 1997
mx_yr = 2020
betterlabels = ['CAP','Unregulated Groundwater','Mixed GW/SW','Regulated Groundwater','Surface Water'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()
# -- For Multiple years --
Name = "Linear Regression during Wet and Normal years for " + data_type
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
dryyrs = [1975,1976,1977
          ,1981,1989,1990
          ,1996,1997,
          1999,2000,2001,2002,2003,2004
          ,2006,2007,2008,2009
          ,2011, 2012, 2013, 2014, 2015, 2016,2017,2018]
wetyrs = [1978,1979,1980,1982,1983,1984,1984,1986,1987,1988
          , 1991,1992,1993,1994,1995,
          1998,2005,2010,2019]

#f = ds[(ds.index == wetyrs)]

f = pd.DataFrame()
for i in wetyrs:
        wut = ds[(ds.index == i)]
        f = f.append(wut)
# print(f)
columns = ds.columns
column_list = ds.columns.tolist()
# ------------------------

stats = pd.DataFrame()
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
m1 = round(stats1.loc['slope','CAP'], 2)
m2 = round(stats1.loc['slope','Unregulated Groundwater'], 2)
m3 = round(stats1.loc['slope','Mixed GW/SW'], 2)
m4 = round(stats1.loc['slope','Regulated Groundwater'], 2)
m5 = round(stats1.loc['slope','Surface Water'], 2)
yint1 = round(stats1.loc['int','CAP'], 2)
yint2 = round(stats1.loc['int','Unregulated Groundwater'], 2)
yint3 = round(stats1.loc['int','Mixed GW/SW'], 2)
yint4 = round(stats1.loc['int','Regulated Groundwater'], 2)
yint5 = round(stats1.loc['int','Surface Water'], 2)
pval1 = round(stats1.loc['p_val', 'CAP'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated Groundwater'], 4)
pval3 = round(stats1.loc['p_val', 'Mixed GW/SW'], 4)
pval4 = round(stats1.loc['p_val', 'Regulated Groundwater'], 4)
pval5 = round(stats1.loc['p_val', 'Surface Water'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

fig, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
ax.plot(xf1, yf3,"-.",color='grey', lw=1)
ax.plot(xf1, yf4,"-.",color='grey', lw=1)
ax.plot(xf1, yf5,"-.",color='grey', lw=1)

f.plot(ax=ax,marker='o', ls='', label=betterlabels)
# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(300,75)
ax.set_title(data_type)
plt.figtext(0.95, 0.5, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98, 0.45, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.4, 'Unreg GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98, 0.35, 'p-value = ' + str(pval2))
plt.figtext(0.95, 0.3, 'Mix equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98, 0.25, 'p-value = ' + str(pval3))
plt.figtext(0.95, 0.2, 'Reg GW (No_CAP) equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98, 0.15, 'p-value = ' + str(pval4))
plt.figtext(0.95, 0.1, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98, 0.05, 'p-value = ' + str(pval5))

ax.legend(loc = [1.065, 0.55])
plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'.csv')

# %% For Depth to Water by regulation
ds = cat_wl2
data_type = "Depth to Water"
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

fig, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
f.plot(ax=ax,marker='o', ls='', label=betterlabels)
ax.set_title(data_type)
plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
# plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/'+Name+'.csv')



# %% ====== Going to run the Spearman's rho coeficient analysis ======
# First read in the drought indice
drought_indices = pd.read_csv('../MergedData/Output_files/Yearly_DroughtIndices.csv')
drought_indices = drought_indices.set_index('In_year')
drought_indices
# %% Water Analysis period
wlanalysis_period = cat_wl2[cat_wl2.index>=1975]
wlanalysis_period

# %%
fig, ax = plt.subplots(1, 1)
ax.scatter(drought_indices['PDSI'], wlanalysis_period['CAP'],color='grey', lw=1)

# %% Test Analysis
wlanalysis_period['PDSI'] = drought_indices['PDSI']
wlanalysis_period
# %% Running the test
# x_simple = pd.DataFrame([(-2,4),(-1,1),(0,3),(1,2),(2,0)],
#                         columns=["X","Y"])
# my_r = x_simple.corr(method="spearman")
# print(my_r)

rho, pval = sp.spearmanr(wlanalysis_period.shift(1), drought_indices['PDSI'])
print("rho = ", rho, '; p-value - ',pval)

# %% Plotting to see if there's a relationship
#%% Plot just the regulated
ds = wlanalysis_period
minyear=1993
maxyear=2020
lag = -4
name = "Average DTW and PDSI from " + str(minyear) + " to " + str(maxyear) + ' lagged by ' + str(lag)
min_y = 100
max_y = 250
fsize = 14

fig, ax = plt.subplots(figsize = (16,9))
ax.plot(ds['R'].shift(lag), label='GW Regulated', color=c_2) 
# ax.plot(ds['U'].shift(lag), label='GW Unregulated', color=c_7)

# Secondary Axis
ax2 = ax.twinx()
ax2.set_ylabel('PDSI')
ax2.set_ylim(-7, 10)
ax2.plot(ds['PDSI'], '-.',label='PDSI', color='grey', lw = 3, zorder=0) 

ax.set_xlim(minyear,maxyear)
ax.set_ylim(max_y,min_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)

ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.05, 0.40], fontsize = fsize)
ax2.legend(loc = [1.05, 0.35], fontsize = fsize)

# # Drought Year Shading
a = 1988.5
b = 1990.5
c = 1995.5
d = 1996.5
e = 2001.5
f = 2003.5
g = 2005.5
h = 2007.5
i = 2011.5
j = 2014.5
k = 2017.5
l= 2018.5
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_byregulation', bbox_inches='tight')
# plt.savefig(outputpath+name+'_byregulation_Drought', bbox_inches='tight')
plt.savefig(outputpath+name+'_GWReg_Drought', bbox_inches='tight')


# %%
def display_correlation(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Spearman Correlation")
    return(r)

def display_corr_pairs(df,color="cyan"):
    s = set_title = np.vectorize(lambda ax,r,rho: ax.title.set_text("r = " + 
                                        "{:.2f}".format(r) + 
                                        '\n $\\rho$ = ' + 
                                        "{:.2f}".format(rho)) if ax!=None else None
                            )      

    r = display_correlation(df)
    rho = df.corr(method="pearson")
    g = sns.PairGrid(df,corner=True)
    g.map_diag(plt.hist,color="yellow")
    g.map_lower(sns.scatterplot,color="magenta")
    set_title(g.axes,r,rho)
    plt.subplots_adjust(hspace = 0.6)
    plt.show()   
# %%
display_correlation(wlanalysis_period)
# %%
display_corr_pairs(wlanalysis_period)

# %%
# For Depth to Water by SW Access
ds = cat_wl2
data_type = "Depth to Water"
min_yr = 1975
mx_yr = 2020
betterlabels = ['CAP','Unregulated Groundwater','Mixed GW/SW','Regulated Groundwater','Surface Water'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()
# # -- For Multiple years --
# Name = "Linear Regression during Wet and Normal years for " + data_type
# # wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# # dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# dryyrs = [1975,1976,1977
#           ,1981,1989,1990
#           ,1996,1997,
#           1999,2000,2001,2002,2003,2004
#           ,2006,2007,2008,2009
#           ,2011, 2012, 2013, 2014, 2015, 2016,2017,2018]
# wetyrs = [1978,1979,1980,1982,1983,1984,1984,1986,1987,1988
#           , 1991,1992,1993,1994,1995,
#           1998,2005,2010,2019]

# #f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in wetyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# # print(f)
# columns = ds.columns
# column_list = ds.columns.tolist()
# ------------------------

stats = pd.DataFrame()
df2 = pd.DataFrame()
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
        trend = (slope*x)+intercept
        y2 = y - trend
        df2[i] = y2
        # print(y2)
        slope2, intercept2, r_value2, p_value2, std_err2 =sp.linregress(x,y2)
        stats = stats.append({'slope': slope2, 
                              'int':intercept2, 
                              'rsq':r_value2*r_value2, 
                              'p_val':p_value2, 
                              'std_err':std_err2, 
                              'mean': np.mean(y2),
                              'var': np.var(y2),
                              'sum': np.sum(y2)
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

df2=df2.set_index(np.array(pd.to_datetime(df).index.values, dtype=float))

stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)

# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
# m1 = round(stats1.loc['slope','CAP'], 2)
# m2 = round(stats1.loc['slope','Unregulated Groundwater'], 2)
# m3 = round(stats1.loc['slope','Mixed GW/SW'], 2)
# m4 = round(stats1.loc['slope','Regulated Groundwater'], 2)
# m5 = round(stats1.loc['slope','Surface Water'], 2)
# yint1 = round(stats1.loc['int','CAP'], 2)
# yint2 = round(stats1.loc['int','Unregulated Groundwater'], 2)
# yint3 = round(stats1.loc['int','Mixed GW/SW'], 2)
# yint4 = round(stats1.loc['int','Regulated Groundwater'], 2)
# yint5 = round(stats1.loc['int','Surface Water'], 2)
m1 = stats1.loc['slope','CAP']
m2 = stats1.loc['slope','Unregulated Groundwater']
m3 = stats1.loc['slope','Mixed GW/SW']
m4 = stats1.loc['slope','Regulated Groundwater']
m5 = stats1.loc['slope','Surface Water']
yint1 = stats1.loc['int','CAP']
yint2 = stats1.loc['int','Unregulated Groundwater']
yint3 = stats1.loc['int','Mixed GW/SW']
yint4 = stats1.loc['int','Regulated Groundwater']
yint5 = stats1.loc['int','Surface Water']

pval1 = round(stats1.loc['p_val', 'CAP'], 4)
pval2 = round(stats1.loc['p_val', 'Unregulated Groundwater'], 4)
pval3 = round(stats1.loc['p_val', 'Mixed GW/SW'], 4)
pval4 = round(stats1.loc['p_val', 'Regulated Groundwater'], 4)
pval5 = round(stats1.loc['p_val', 'Surface Water'], 4)

yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

fig, ax = plt.subplots(1, 1)
ax.plot(xf1, yf1,"-.",color='grey',label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color='grey', lw=1)
ax.plot(xf1, yf3,"-.",color='grey', lw=1)
ax.plot(xf1, yf4,"-.",color='grey', lw=1)
ax.plot(xf1, yf5,"-.",color='grey', lw=1)

# f.plot(ax=ax,marker='o', ls='', label=betterlabels)
df2.plot(ax=ax,marker='o', ls='', label=betterlabels)

# ax.set_xlim(min_yr, mx_yr)
# ax.set_ylim(300,75)
ax.set_title(data_type)
plt.figtext(0.95, 0.5, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98, 0.45, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.4, 'Unreg GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98, 0.35, 'p-value = ' + str(pval2))
plt.figtext(0.95, 0.3, 'Mix equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98, 0.25, 'p-value = ' + str(pval3))
plt.figtext(0.95, 0.2, 'Reg GW (No_CAP) equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98, 0.15, 'p-value = ' + str(pval4))
plt.figtext(0.95, 0.1, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98, 0.05, 'p-value = ' + str(pval5))

ax.legend(loc = [1.065, 0.55])

# Subtract off the trend


# plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'.csv')

# %%
df2.plot()

# %%
#%% Plot by access to surfacewater
ds = df2
minyear=1975
maxyear=2020
name = "Average Depth to Water minus the trend " + str(minyear) + " to " + str(maxyear) + " by Access to SW"
min_y = 0
max_y = 300
fsize = 14

fig, ax = plt.subplots(figsize = (16,9))
ax.plot(ds['CAP'], label='CAP', color=c_2)
ax.plot(ds['No_CAP'], label='Regulated GW', color=c_3) 
ax.plot(ds['GW'], label='Unregulated GW', color=c_7) 
ax.plot(ds['Mix'], label='Mixed SW/GW', color=c_5) 
ax.plot(ds['SW'], label='Surface Water', color=c_4) 

ax.set_xlim(minyear,maxyear)
# ax.set_ylim(max_y,min_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Water Level (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
# Drought Year Shading
a = 1988.5
b = 1989.5
c = 1995.5
d = 1996.5
# e = 1999.5
# f = 2000.5
g = 2001.5
h = 2003.5
i = 2005.5
j = 2007.5
k = 2011.5
l = 2014.5
m = 2017.5
n = 2018.5
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(m, n, color=drought_color, alpha=0.5, lw=0)

# # Wet years (2005 and 2010)
# g = 2005
# h = 2010
# ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
# ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)
ax.minorticks_on()

fig.set_dpi(600.0)

plt.savefig(outputpath+name+'_Drought', bbox_inches='tight')
# plt.savefig(outputpath+name+'_byGW', bbox_inches='tight')
# plt.savefig(outputpath+name+'_bySW', bbox_inches='tight')
# plt.savefig(outputpath+name, bbox_inches='tight')

# %% === Shifted correlation analysis ===


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
