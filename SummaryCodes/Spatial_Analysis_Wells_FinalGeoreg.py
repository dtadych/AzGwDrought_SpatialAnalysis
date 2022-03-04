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

# %%
# Load in the master database

# Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database.shp')
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/Final_Georegions/'
# %%
filename = 'Master_ADWR_database_noduplicates.shp'
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
print("join complete")
# %% Exporting it because I guess I did that before since I load it in later
static_geo.to_csv('../MergedData/Output_files/Final_Static_geodatabase_allwells.csv')

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

#%% Plotting
ds = cat_wl2
minyear=1970
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 0
max_y = 350

# Plot all of them
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
#static_geo2 = static_geo
#static_geo2.info()

# %%
static_geo2['APPROVED'] = pd.to_datetime(static_geo2['APPROVED'])
static_geo2['APPROVED'].describe()
# %%
static_geo2['INSTALLED'] = pd.to_datetime(static_geo2['INSTALLED'])
static_geo2['INSTALLED'].describe()
# %%
static_geo2['In_year'] = static_geo2['INSTALLED'].dt.year

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
wdc1 = pd.pivot_table(wd1, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc2 = pd.pivot_table(wd2, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)
wdc3 = pd.pivot_table(wd3, index=["In_year"], columns=["GEO_Region"], values=['WELL_DEPTH'], dropna=False, aggfunc=len)

#%% Exporting the Depth categories
#wdc1.to_csv('../MergedData/Output_files/Final_Welldepth' + str(deep) + 'plus.csv')
#wdc2.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'to' + str(deep) + '.csv')
#wdc3.to_csv('../MergedData/Output_files/Final_Welldepth' + str(shallow) + 'minus.csv')

# %% Plotting fun
columns = wdc1.columns
columns

# %%
for i in columns:
        stuff = wdc1[i].rename(labels)

print(stuff)
# %%
ds = wdc1
ds.reset_index
#labels = ds.columns.tolist()
#labels

fig, ax = plt.subplots()
for i in labels:
        ax.plot(ds[i], label = labels[i])
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
ds = new_wells
ds.reset_index
#labels = ds.columns.tolist()
#labels

fig, ax = plt.subplots()
for i in labels:
        ax.plot(ds[i], label = labels[i])
ax.set(title='Number of new wells per region', xlabel='Year', ylabel='Well Depth (ft)'
       , xlim = [1980,2020]
        )
#ax.xaxis.set_major_locator(cat_wl2.Final_Region(interval=50))
#ax.set_xticklabels()
ax.legend(loc = [1.05, 0])
# %%
georeg['area'] = georeg.geometry.area
georeg
# %%
georeg2 = pd.DataFrame(georeg)
georeg2
# %%
del georeg2['geometry']
georeg2.info()
# %%
georeg2.to_csv('../MergedData/Output_files/georegions_area.csv')

# %%
# %% Plotting help from Amanda - don't run this
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