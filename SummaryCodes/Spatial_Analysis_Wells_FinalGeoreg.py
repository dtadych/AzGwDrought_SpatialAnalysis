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

# %%
# Load in the master database

# Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database.shp')
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/Final_Georegions/'
# %%
filename = 'Master_ADWR_database_water.shp'
filepath = os.path.join(outputpath, filename)
print(filepath)

masterdb = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb.info())
# %%
# Reading in the shapefile
# GEOREG.to_file('../MergedData/Output_files/Georegions_3col.shp')
filename = "Final_Georegions.shp"
filepath = os.path.join(shapepath, filename)
georeg = gp.read_file(filepath)
# %%
#georeg.boundary.plot()
georeg.plot(cmap='viridis').legend()

#%%
georeg['GEOREGI_NU'] = georeg['GEOREGI_NU'].astype('int64')
georeg.info()
#%%
# Read in the annual time series database
filename = 'Wells55_GWSI_WLTS_DB_annual.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)
# %%
annual_db = pd.read_csv(filepath, header=1, index_col=0)
annual_db.head()
#pd.options.display.float_format = '{:.2f}'.format
#%%
annual_db.index.astype('int64')
#%%
annual_db.head()

# %% Overlay georegions onto the static database
# Going to use sjoin based off this website: https://geopandas.org/docs/user_guide/mergingdata.html
print(masterdb.crs, georeg.crs)

# %%
georeg = georeg.to_crs(epsg=26912)
# %%
static_geo = gp.sjoin(masterdb, georeg, how="inner", op='intersects')
static_geo.head()

# %% Exporting it because I guess I did that before since I load it in later
#static_geo.to_csv('../MergedData/Output_files/Final_Static_geodatabase.csv')

# %% Create a dataframe of Final_Region and Well ID's
reg_list = static_geo[['Combo_ID', 'GEO_Region']]
reg_list

# %% Converting Combo_ID to int
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
cat_wl = combo.groupby(['GEO_Region']).mean()
cat_wl

# %%
cat_wl2 = cat_wl.transpose()

# %% Going to export all these as CSV's
#cat_wl.to_csv('../MergedData/Output_files/Final_Categories_WL.csv')
#combo.to_csv('../MergedData/Output_files/Final_WaterLevels.csv')
#cat_wl2.to_csv('../MergedData/Output_files/Final_WaterLevels_transposedforgraphing.csv')


# %% Trying to fix the datetime issue but it's not working.  Just skip to line 172
cat_wl2.reset_index(inplace=True)
cat_wl2
# %%
cat_wl2.info()
# %%
cat_wl2['index'] = cat_wl2['index'].astype(int)
cat_wl2.info()
#%%
cat_wl2.index = pd.to_datetime(cat_wl2.index)
cat_wl2.head()
# %% 
cat_wl2.index = pd.DatetimeIndex(cat_wl2.index)
# %%
cat_wl2.plot()
#%% Plotting
fig, ax = plt.subplots()
ax.plot(cat_wl2['Reg_CAP'], label='Regulated - CAP') 
ax.plot(cat_wl2['Reg_CAP_Other'], label='Regulated CAP and Other SW')
ax.plot(cat_wl2['Reg_NoSW'], label='Regulated - No SW')
ax.plot(cat_wl2['Res'], label='Reservations')
ax.plot(cat_wl2['Unreg_CoR'], label='Unregulated - Colorado River')
ax.plot(cat_wl2['Unreg_NoSW'], label='Unregulated - No SW')
ax.plot(cat_wl2['Unreg_Other'], label='Unregulated - Other SW')
ax.set_xlim(0,152)
ax.set(title='Average depth to Water since 1853', xlabel='Year', ylabel='Water Level (ft)')
ax.legend(loc = [1.05, 0.50])

# %% 1950 - present day
fig, ax = plt.subplots()
ax.plot(cat_wl2['Reg_CAP'], label='Regulated - CAP')
ax.plot(cat_wl2['Reg_CAP_Other'], label='Regulated CAP and Other SW')
ax.plot(cat_wl2['Reg_NoSW'], label='Regulated - No SW')
ax.plot(cat_wl2['Res'], label='Reservations')
ax.plot(cat_wl2['Unreg_CoR'], label='Unregulated - Colorado River')
ax.plot(cat_wl2['Unreg_NoSW'], label='Unregulated - No SW')
ax.plot(cat_wl2['Unreg_Other'], label='Unregulated - Other SW')
#ax.set_xlim(100,155)
ax.set(title='Average depth to Water since 1950', xlabel='Year', ylabel='Water Level (ft)',
        xlim = [40, 155])
#ax.xaxis.set_major_locator(cat_wl2.Final_Region(interval=50))
#ax.set_xticklabels()
ax.legend(loc = [1.05, 0.50])

# %% --- Now making graphs for other things
# - Well Density (cumulative number of wells) over time
# - max screen depth over time (Casing_DEP vs Installed)
# - Number of new wells installed over time

# Re-read in after proper formatting
filename = 'Final_Static_geodatabase.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)
static_geo2 = pd.read_csv(filepath 
                          ,parse_dates=['INSTALLED']
                          )
static_geo2

# %% 
static_geo2 = static_geo
static_geo2.info()

# %%
static_geo2['APPROVED'] = pd.to_datetime(static_geo2['APPROVED'])
static_geo2['APPROVED'].describe()
# %%
static_geo2['INSTALLED'] = pd.to_datetime(static_geo2['INSTALLED'])
static_geo2['INSTALLED'].describe()
# %%
static_geo2['In_year'] = static_geo2['INSTALLED'].dt.year

# %% 
Well_Depth = static_geo2[['WELL_DEPTH', 'INSTALLED', 'Combo_ID', 'In_year','GEO_Region']]
#%%
Well_Depth

#%%
# Well_Depth.to_csv('../MergedData/Output_files/Final_WellDepth.csv')
#%%
Well_Depth = pd.pivot_table(static_geo2, index=["In_year"], columns=["GEO_Region"], values=["WELL_DEPTH"], dropna=False, aggfunc=np.mean)
Well_Depth.describe()

# %% Set shallow and drilling depths
shallow = 200
deep = 500

# %%
wd1 = static_geo2[(static_geo2["WELL_DEPTH"] > deep)]
wd2 = static_geo2[(static_geo2["WELL_DEPTH"] <= deep) & (static_geo2["WELL_DEPTH"] >= shallow)]
wd3 = static_geo2[(static_geo2["WELL_DEPTH"] < shallow)]

# %%
wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

#%% Exporting the Depth categories
#wdc1.to_csv('../MergedData/Output_files/Final_Welldepth' + str(deep) + 'plus.csv')
#wdc2.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'to' + str(deep) + '.csv')
#wdc3.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'minus.csv')

# %%
ds = wdc1
ds.reset_index
labels = ds.columns.tolist()
labels

fig, ax = plt.subplots()
for i in labels:
        ax.plot(ds[i], label = i)
ax.set(title='Number of Shallow Wells (less than '+ str(shallow) +')'
        , xlabel='Year', ylabel='Well Depth (ft)'
        , xlim = [1980,2020]
        )
#ax.xaxis.set_major_locator(cat_wl2.Final_Region(interval=50))
#ax.set_xticklabels()
ax.legend(loc = [1.05, 0])
# %%
ds = wdc2
ds.reset_index
labels = ds.columns.tolist()
labels

fig, ax = plt.subplots()
for i in labels:
        ax.plot(ds[i], label = i)
ax.set(title='Number of Mid-range Wells (between '+ str(shallow) +' and '+ str(deep)+')', xlabel='Year', ylabel='Well Depth (ft)'
       , xlim = [1980,2020]
        )
#ax.xaxis.set_major_locator(cat_wl2.Final_Region(interval=50))
#ax.set_xticklabels()
ax.legend(loc = [1.05, 0])

# %%
ds = wdc3
ds.reset_index
labels = ds.columns.tolist()
labels

fig, ax = plt.subplots()
for i in labels:
        ax.plot(ds[i], label = i)
ax.set(title='Number of Deep Wells (greater than '+ str(deep) +')', xlabel='Year', ylabel='Well Depth (ft)'
       , xlim = [1980,2020]
        )
#ax.xaxis.set_major_locator(cat_wl2.Final_Region(interval=50))
#ax.set_xticklabels()
ax.legend(loc = [1.05, 0])

# %%
static_geo2['INSTALLED'] = static_geo2['INSTALLED'].replace(['1970-01-01T00:00:00.000000000'], np.NaN)

# %%
new_wells = pd.pivot_table(static_geo2, index=["In_year"], columns=["GEO_Region"], values=["INSTALLED"], dropna=False, aggfunc=len)
new_wells
# %%
new_wells.to_csv('../MergedData/Output_files/Final_NewWells.csv')

# %%
fig, ax = plt.subplots()
ax.plot(new_wells['Reg_CAP'], label='Regulated - CAP')
ax.plot(new_wells['Reg_CAP_Other'], label='Regulated CAP and Other SW')
ax.plot(new_wells['Reg_NoSW'], label='Regulated - No SW')
ax.plot(new_wells['Res'], label='Reservations')
ax.plot(new_wells['Unreg_CoR'], label='Unregulated - Colorado River')
ax.plot(new_wells['Unreg_NoSW'], label='Unregulated - No SW')
ax.plot(new_wells['Unreg_Other'], label='Unregulated - Other SW')
#ax.set_xlim(100,155)
ax.set(title='Average Borehole Depth since 1950', xlabel='Year', ylabel='Number of new wells')
#ax.xaxis.set_major_locator(cat_wl2.Final_Region(interval=50))
#ax.set_xticklabels()
ax.legend(loc = [1.05, 0.50])
# %%
