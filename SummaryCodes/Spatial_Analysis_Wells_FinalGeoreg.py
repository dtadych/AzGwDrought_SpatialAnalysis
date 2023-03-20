# The purpose of this script is to create a code to spatially analyze all the wells in 
# the combined database based on management. 
# Written by Danielle Tadych
# Goals:
# - create columns with the different management regions
#       1. AMA/INA
#       2. Irrigation District
#       3. AI Homelands
#       4. Unregulated

# Potential workflow
# - import master database and shape files
# - Create columns of the different management options (did this in QGIS)
# - Create an if statement of if a well falls within a certain region, then it can equal 
#       the name of the shape it falls under

# WORKFLOW THAT ACTUALLY HAPPENED
# 1. Read in the master ADWR database static database, water level database, and 
#       georegions shapefile created in QGIS
# 2. Overlayed region shapefile on static well database shapefile
# 3. Exported a dataframe (registry list) of combined ID's with the columns we want 
#       (regulation, etc.)
# 4. Joined the registry list with the timeseries database so every well has water 
#       levels and is tagged with a category we want
# 5. Create pivot tables averaging water levels based on categories (e.g. regulation, 
#       access to SW, or georegion (finer scale))
# 6. Export pivot tables into .csv's for easy analyzing later
#       * Note: after reading in packages, skip to line 197 to avoid redoing steps 1-5
# 7. Graphs for days (starting around line 214)
# 8. Statistical analyses
#       - Linear Regression (~line 929)
#       - Pearson/Spearman Correlation (~line 212)
#       - lagged Correlation analyses


# %%
from cProfile import label
from curses import nocbreak
# from dbm import _ValueType
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
from scipy.stats import kendalltau, pearsonr, spearmanr
import pymannkendall as mk


# Some functions for analysis
def kendall_pval(x,y):
        return kendalltau(x,y)[1]
    
def pearsonr_pval(x,y):
        return pearsonr(x,y)[1]
    
def spearmanr_pval(x,y):
        return spearmanr(x,y)[1]

def display_correlation(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(df.corr(method='spearman'), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Spearman Correlation")
    return(r)

def display_corr_pairs(df,color="cyan"):
    s = set_title = np.vectorize(lambda ax,r,rho: ax.title.set_text("r = " + 
                                        "{:.2f}".format(r) + 
                                        '\n $\\rho$ = ' + 
                                        "{:.2f}".format(rho)) if ax!=None else None
                            )      

    rho = display_correlation(df)
    r = df.corr(method="pearson")
    g = sns.PairGrid(df,corner=True)
    g.map_diag(plt.hist,color="yellow")
    g.map_lower(sns.scatterplot,color="magenta")
    set_title(g.axes,r,rho)
    plt.subplots_adjust(hspace = 0.6)
    plt.show()  

# Data paths
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/Final_Georegions/'

# %%  Load in the master databases
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
# filename_georeg = "Final_Georegions.shp"
filename_georeg = 'georeg_reproject_fixed.shp'
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
filename = "Final_Static_geodatabase_allwells.csv"
filepath = os.path.join(outputpath, filename)
static_geo = pd.read_csv(filepath)
static_geo

# %% Create a dataframe of Final_Region and Well ID's
reg_list = static_geo[['Combo_ID', 'GEO_Region', 'GEOREGI_NU','Water_CAT', 'Loc','Regulation','WELL_DEPTH']]
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

# %% Exporting the combo table
# combo.to_csv('../MergedData/Output_files/Final_WaterLevels_adjusted.csv')

# %% Reading in so we don't have to redo the combining
filepath = '../MergedData/Output_files/Final_WaterLevels_adjusted.csv'
combo = pd.read_csv(filepath, index_col=0)
combo.head()

# %% in order to filter deep/mid/shallow wells
shallow = 200
deep = 500

wd1 = combo[(combo["WELL_DEPTH"] > deep)]
wd2 = combo[(combo["WELL_DEPTH"] <= deep) & (combo["WELL_DEPTH"] >= shallow)]
wd3 = combo[(combo["WELL_DEPTH"] < shallow)]

# %% in order to make it where we can actually group these bitches
whatever = [combo,wd1,wd2,wd3]
for i in whatever:
        del i['WELL_DEPTH']

# %% Now for aggregating by category for the timeseries
# to narrow by depth database
# combo = wd1

cat_wl_georeg = combo.groupby(['GEOREGI_NU']).mean()
cat_wl_reg = combo.groupby(['Regulation']).mean()
cat_wl_SW = combo.groupby(['Water_CAT']).mean()

cat_wl_georeg.info()

# %%
wdc1_reg = wd1.groupby(['Regulation']).mean() # deep
wdc2_reg = wd2.groupby(['Regulation']).mean() # midrange
wdc3_reg = wd3.groupby(['Regulation']).mean() # shallow

wdc1_SW = wd1.groupby(['Water_CAT']).mean()
wdc2_SW = wd2.groupby(['Water_CAT']).mean()
wdc3_SW = wd3.groupby(['Water_CAT']).mean()

# %%
i = wdc1_reg
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc1_reg = f

i = wdc2_reg
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc2_reg = f

i = wdc3_reg
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc3_reg = f

i = wdc1_SW
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc1_SW = f

i = wdc2_SW
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc2_SW = f

i = wdc3_SW
i = i.sort_values(by=['GEOREGI_NU'])
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
wdc3_SW = f
# %% 
cat_wl2_georeg = cat_wl_georeg.copy()
cat_wl2_reg = cat_wl_reg.copy()
cat_wl2_SW = cat_wl_SW.copy()

cat_wl2_georeg = cat_wl2_georeg.sort_values(by=['GEOREGI_NU'])
cat_wl2_SW = cat_wl2_SW.sort_values(by=['GEOREGI_NU'])

# Clean up the dataframe for graphing

# databases = [cat_wl2_georeg,cat_wl2_reg,cat_wl2_SW]
# for i in databases:
#         # wlanalysis_period = cat_wl2[cat_wl2.index>=1975]
#         f = i.transpose()
#         # print(i)
#         f.reset_index(inplace=True)
#         f['index'] = pd.to_numeric(f['index'])
#         f['index'] = f['index'].astype(int)
#         f.set_index('index', inplace=True)
#         f = f[f.index>=1975]
#         i = f
#         print(i.describe())

i = cat_wl2_georeg
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
cat_wl2_georeg = f
        
i = cat_wl2_reg
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
cat_wl2_reg = f

i = cat_wl2_SW
del i['GEOREGI_NU']
f = i.transpose()
f.reset_index(inplace=True)
f['index'] = pd.to_numeric(f['index'])
f['index'] = f['index'].astype(int)
f.set_index('index', inplace=True)
f.info()
cat_wl2_SW = f
# %% Going to export all these as CSV's
# cat_wl2_georeg.to_csv('../MergedData/Output_files/Waterlevels_georegions.csv')
# cat_wl2_reg.to_csv('../MergedData/Output_files/Waterlevels_Regulation.csv')
# cat_wl2_SW.to_csv('../MergedData/Output_files/Waterlevels_Waterlevels_AccesstoSW.csv')

# %%  ==== Reading in the data we created above ====
# For regulation
filepath = '../MergedData/Output_files/Waterlevels_Regulation.csv'
cat_wl2_reg = pd.read_csv(filepath, index_col=0)
cat_wl2_reg.head()

# For Access to SW
filepath = '../MergedData/Output_files/Waterlevels_AccesstoSW.csv'
cat_wl2_SW = pd.read_csv(filepath, index_col=0)
cat_wl2_SW.head()

# For georegion number
filepath = '../MergedData/Output_files/Waterlevels_georegions.csv'
cat_wl2_georeg = pd.read_csv(filepath, index_col=0)
# cat_wl2_georeg.head()
# %%
cat_wl2_georeg = cat_wl2_georeg.transpose()
cat_wl2_georeg
# %%
cat_wl2_georeg.reset_index(inplace=True)
cat_wl2_georeg['index'] = pd.to_numeric(cat_wl2_georeg['index'])
cat_wl2_georeg.set_index('index', inplace=True)
cat_wl2_georeg.info()

# %%
cat_wl2_georeg = cat_wl2_georeg.transpose()
cat_wl2_georeg

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

reg_colors = [c_2,c_7]
georeg_colors = [c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11]
SW_colors = [c_2,c_3,c_4,c_5,c_7]

bar_watercatc = [c_2,c_3,c_4,c_5,c_7]

# Color blind palette
# https://jacksonlab.agronomy.wisc.edu/2016/05/23/15-level-colorblind-friendly-palette/
blind =["#000000","#004949","#009292","#ff6db6","#ffb6db",
 "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
 "#920000","#924900","#db6d00","#24ff24","#ffff6d"]

# Matching new map

cap = '#C6652B'
# noCAP = '#EDE461' # This is one from the map
noCAP = '#CCC339' # This color but darker for lines
GWdom = '#3B76AF'
mixed = '#6EB2E4'
swdom = '#469B76'


# %% DTW by well depths and Access to SW
# ds = wdc1.copy()
name = 'Average Depth to water for Access to SW Categories'
ds = wdc1_SW.copy()
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
ds = wdc2_SW.copy()
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
ds = wdc3_SW.copy()
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

df1 = pd.DataFrame(dft1.mean())
df1 = df1.transpose()
df1 = df1.reset_index()
df1['index'] = 'Deep'
df1.set_index('index', inplace=True)
df1

df2 = pd.DataFrame(dft2.mean())
df2 = df2.transpose()
df2 = df2.reset_index()
df2['index'] = 'Midrange'
df2.set_index('index', inplace=True)
df2

df3 = pd.DataFrame(dft3.mean())
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
horlabel = 'Depth to water (ft)'
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
plt.xticks(rotation=30)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(fontsize = fsize)

# plt.savefig(outputpath+name+'groupedchart')


#%% Plot by Groundwater Regulation
# ds = cat_wl2_reg
ds = wdc3_reg #shallow
# ds = wdc2_reg #midrange
# ds = wdc1_reg #deep 
minyear=1975
maxyear=2020
# name = "Average Depth to Water from " + str(minyear) + " to " + str(maxyear) + ' by Groundwater Regulation'
# name = "Average Depth to Water for Deep Wells by Groundwater Regulation" # wd1
# name = "Average Depth to Water for Midrange Wells by Groundwater Regulation" # wd2
name = "Average Depth to Water for Shallow Wells by Groundwater Regulation" #wd3

min_y = 75
max_y = 300
fsize = 14

fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds['R'], label='GW Regulated', color=c_2) 
ax.plot(ds['U'], label='GW Unregulated', color=c_7) 
ax.set_xlim(minyear,maxyear)
# ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
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
ax.minorticks_on()


fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_byregulation', bbox_inches='tight')
# plt.savefig(outputpath+name+'_timeseries_Drought', bbox_inches='tight')


#%% Plot by access to surfacewater
ds = wdc1_SW
minyear=1975
maxyear=2020
# name = "Average Depth to Water from " + str(minyear) + " to " + str(maxyear) + " by Access to SW"
# name = "Average Depth to Water for Deep Wells by Access to SW" # wd1
# name = "Average Depth to Water for Midrange Wells by Access to SW" # wd2
# name = "Average Depth to Water for Deep Wells by Access to SW" # wd3
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
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
# Drought Year Shading
# a = 1988.5
# b = 1990.5
# c = 1995.5
# d = 1996.5
# e = 2001.5
# f = 2003.5
# g = 2005.5
# h = 2007.5
# i = 2011.5
# j = 2014.5
# k = 2017.5
# l= 2018.5
# plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

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
# plt.savefig(outputpath+name, bbox_inches='tight')

#%% Plotting individual divisions
ds = cat_wl2_georeg
minyear=1975
maxyear=2020
# name = "Average Depth to Water from " + str(minyear) + " to " + str(maxyear)
name = "Average Depth to Water for Deep Wells by Georegion" # wd1
# name = "Average Depth to Water for Midrange Wells by Georegion" # wd2
# name = "Average Depth to Water for Deep Wells by Georegion" # wd3

min_y = 150
max_y = 600
fsize = 14

# Plot all of them on a single graph
fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds[2.0], label='Regulated with CAP', color=c_2) 
ax.plot(ds[3.0], label='Regulated without CAP', color=c_3) 
ax.plot(ds[4.0], color=c_4, label='Lower Colorado River - SW Dominated')
ax.plot(ds[5.0], color=c_5, label='Upper Colorado River - Mixed')
# ax.plot(ds[10.0], color=c_10, label='North - Mixed')
ax.plot(ds[11.0], color=c_11, label='Central - Mixed')
ax.plot(ds[7.0], color=c_7, label='Northwest - GW Dominated')
ax.plot(ds[9.0], color=c_9, label='Northeast - GW Dominated')
ax.plot(ds[8.0], color=c_8, label='South central - GW Dominated')
ax.plot(ds[6.0], color=c_6, label='Southeast - GW Dominated')
# Drought Year Shading
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

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
ax.grid(True)
ax.set(title=name, xlabel='Year', ylabel='Depth to Water (ft)')
ax.legend(loc = [1.04, 0.40])

#%% Plot just the regulated
fig, ax = plt.subplots(figsize = (16,9))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
ax.plot(ds[2.0], label='Regulated with CAP', color=c_2) 
ax.plot(ds[3.0], label='Regulated without CAP', color=c_3) 
ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
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
ax.set_ylim(min_y,max_y)
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
ax.set_ylim(min_y,max_y)
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
# ds = cat_wl2_georeg
# minyear=1970
# maxyear=2020
# name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
# min_y = 150
# max_y = 400
# fsize = 14
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
ax[0,0].set_ylim(min_y,max_y)
ax[0,1].set_ylim(min_y,max_y)
ax[1,0].set_ylim(min_y,max_y)
ax[1,1].set_ylim(min_y,max_y)
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
# ds = cat_wl2_georeg
# minyear=1975
# maxyear=2020
# name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 150
max_y = 600
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
ax[p1].set_ylim(min_y,max_y)
ax[p2].set_ylim(min_y,max_y)
ax[p3].set_ylim(min_y,max_y)
ax[p4].set_ylim(min_y,max_y)
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
# plt.savefig(outputpath+name+'_4panel_1col', bbox_inches = 'tight') # bbox_inches makes sure the legend saves
fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_3panel_drought')
# %% Plot in a four panel 1 column graph
# ds = cat_wl2_georeg
# minyear=2002
# maxyear=2020
# name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 150
max_y = 600
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
ax[p1].set_ylim(min_y,max_y)
ax[p2].set_ylim(min_y,max_y)
ax[p3].set_ylim(min_y,max_y)
ax[p4].set_ylim(min_y,max_y)
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
# plt.savefig(outputpath+name+'_4panel_drought', bbox_inches = 'tight') # bbox_inches makes sure the legend saves

# plt.savefig(outputpath+name+'_3panel_drought')

# %% Plot georegions in a three panel graph, 1 column
ds = cat_wl2_georeg
minyear=1971
maxyear=2020
name = "Average Depth to Water for " + str(minyear) + " to " + str(maxyear)
min_y = 150
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
ax[p1].set_ylim(min_y,max_y)
ax[p2].set_ylim(min_y,max_y)
ax[p3].set_ylim(min_y,max_y)
ax[p4].set_ylim(min_y,max_y)
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
static_geo2 = static_geo
static_geo2.info()

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
# del georeg_area_reg['NA']
georeg_area_reg

# %%
georeg_area_watercat = pd.pivot_table(georeg, columns=["Water_CAT"], values=["area"], dropna=False, aggfunc=np.sum)
# del georeg_area_watercat['NA']
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
ds = cat_wl2_georeg
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
        ax.set_ylim(0,max(y))
        ax.legend()


# stats = stats.append(slope)
#        stats[i] = stats[i].append(slope)

#   df = df.append({'A': i}, ignore_index=True)
stats1 = stats.transpose()
stats1

# %% Linear Regression
# For Depth to Water by SW Access
ds = cat_wl2_SW
data_type = "Depth to Water"
min_yr = 1975
mx_yr = 2020
betterlabels = ['Res','Recieves CAP (Regulated)'
                ,'GW Dominated (Regulated)'
                ,'Surface Water Dominated'
                ,'GW Dominated'
                ,'Mixed Source'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()
# -- For Multiple years --
# Name = "Linear Regression during Wet and Normal years for " + data_type
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# dryyrs = [1975,1976,1977
#           ,1981,1989,1990
#           ,1996,1997,
#           1999,2000,2001,2002,2003,2004
#           ,2006,2007,2008,2009
#           ,2011, 2012, 2013, 2014, 2015,2017,2018]
# wetyrs = [1978,1979,1980,1982,1983,1984,1984,1986,1987,1988
#           , 1991,1992,1993,1994,1995,
#           1998,2005,2010,2019]

#f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in wetyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# print(f)
columns = ds.columns
column_list = ds.columns.tolist()
# ------------------------

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
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


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)
# -- Data visualization --
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
m1 = round(stats1.loc['slope',betterlabels[1]], 2)
m2 = round(stats1.loc['slope',betterlabels[4]], 2)
m3 = round(stats1.loc['slope',betterlabels[5]], 2)
m4 = round(stats1.loc['slope',betterlabels[2]], 2)
m5 = round(stats1.loc['slope',betterlabels[3]], 2)
yint1 = round(stats1.loc['int',betterlabels[1]], 2)
yint2 = round(stats1.loc['int',betterlabels[4]], 2)
yint3 = round(stats1.loc['int',betterlabels[5]], 2)
yint4 = round(stats1.loc['int',betterlabels[2]], 2)
yint5 = round(stats1.loc['int',betterlabels[3]], 2)
rsq1 = round(stats1.loc['rsq',betterlabels[1]], 4)
rsq2 = round(stats1.loc['rsq',betterlabels[4]], 4)
rsq3 = round(stats1.loc['rsq',betterlabels[5]], 4)
rsq4 = round(stats1.loc['rsq',betterlabels[2]], 4)
rsq5 = round(stats1.loc['rsq',betterlabels[3]], 4)
pval1 = round(stats1.loc['p_val',betterlabels[1]], 4)
pval2 = round(stats1.loc['p_val',betterlabels[4]], 4)
pval3 = round(stats1.loc['p_val',betterlabels[5]], 4)
pval4 = round(stats1.loc['p_val',betterlabels[2]], 4)
pval5 = round(stats1.loc['p_val',betterlabels[3]], 4)
yf1 = (m1*xf)+yint1
yf2 = (m2*xf)+yint2
yf3 = (m3*xf)+yint3
yf4 = (m4*xf)+yint4
yf5 = (m5*xf)+yint5

fig, ax = plt.subplots(1, 1, figsize = (7,4.5))
# fig, ax = plt.subplots(figsize = (16,9))

ax.plot(xf1, yf1,"-.",color=cap,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)
ax.plot(xf1, yf3,"-.",color=mixed, lw=1)
ax.plot(xf1, yf4,"-.",color='#CCC339', lw=1)
ax.plot(xf1, yf5,"-.",color=swdom, lw=1)

# f.plot(ax=ax,marker='o', ls='', label=betterlabels)
# Trying to draw lines with better shit 

ds = cat_wl2_SW
minyear=1975
maxyear=2020
min_y = 75
max_y = 300
fsize = 12

ax.plot(ds['CAP'], label=betterlabels[1], color=cap)
ax.plot(ds['No_CAP'], label=betterlabels[2], color='#CCC339') 
ax.plot(ds['SW'], label=betterlabels[3], color=swdom) 
ax.plot(ds['Mix'], label=betterlabels[5], color=mixed)
ax.plot(ds['GW'], label=betterlabels[4], color=GWdom)  

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
# ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = 10)
# # Drought Year Shading
# a = 1988.5
# b = 1990.5
# c = 1995.5
# d = 1996.5
# e = 2001.5
# f = 2003.5
# g = 2005.5
# h = 2007.5
# i = 2011.5
# j = 2014.5
# k = 2017.5
# l= 2018.5
# plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

ax.minorticks_on()

fig.set_dpi(600.0)

# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(75,300)
# ax.set_title(Name)
vertshift = 0
plt.figtext(0.95, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))

ax.legend(
        loc = [1.065, 0.65]
        )
plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'.csv')

# %% Piecewise Linear Regression
# For Depth to Water by SW Access
ds = cat_wl2_SW
data_type = "Depth to Water"
# -- Piece 1 --
min_yr = 1975
mx_yr = 1985
betterlabels = ['Res','Recieves CAP (Regulated)'
                ,'GW Dominated (Regulated)'
                ,'Surface Water Dominated'
                ,'GW Dominated'
                ,'Mixed Source'] 
Name1 = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name1)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
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
rsq1 = round(stats1.loc['rsq', 'CAP'], 4)
rsq2 = round(stats1.loc['rsq', 'Unregulated Groundwater'], 4)
rsq3 = round(stats1.loc['rsq', 'Mixed GW/SW'], 4)
rsq4 = round(stats1.loc['rsq', 'Regulated Groundwater'], 4)
rsq5 = round(stats1.loc['rsq', 'Surface Water'], 4)
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

fig, ax = plt.subplots(1, 1, figsize = (12,7))
# fig, ax = plt.subplots(figsize = (16,9))

# ax.plot(xf1, yf1,"-.",color=c_2,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=c_7, lw=1)
ax.plot(xf1, yf3,"-.",color=c_5, lw=1)
ax.plot(xf1, yf4,"-.",color=c_3, lw=1)
# ax.plot(xf1, yf5,"-.",color=c_4, lw=1)

vertshift = -0.3
horshift = 0
plt.figtext(0.94+horshift, 0.55 - vertshift, 'Regression for ' +str(min_yr)+' to '+str(mx_yr)+':')
plt.figtext(0.95+horshift, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98+horshift, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95+horshift, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98+horshift, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95+horshift, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98+horshift, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95+horshift, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98+horshift, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95+horshift, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98+horshift, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))

# -- Piece 2 --
min_yr = 1985
mx_yr = 1995
Name2 = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name2)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
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
rsq1 = round(stats1.loc['rsq', 'CAP'], 4)
rsq2 = round(stats1.loc['rsq', 'Unregulated Groundwater'], 4)
rsq3 = round(stats1.loc['rsq', 'Mixed GW/SW'], 4)
rsq4 = round(stats1.loc['rsq', 'Regulated Groundwater'], 4)
rsq5 = round(stats1.loc['rsq', 'Surface Water'], 4)
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

# ax.plot(xf1, yf1,"-.",color=c_2, lw=1)
ax.plot(xf1, yf2,"-.",color=c_7, lw=1)
ax.plot(xf1, yf3,"-.",color=c_5, lw=1)
ax.plot(xf1, yf4,"-.",color=c_3, lw=1)
# ax.plot(xf1, yf5,"-.",color=c_4, lw=1)

vertshift = -0.3
horshift = 0.3
plt.figtext(0.94+horshift, 0.55 - vertshift, 'Regression for ' +str(min_yr)+' to '+str(mx_yr)+':')
plt.figtext(0.95+horshift, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98+horshift, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95+horshift, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98+horshift, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95+horshift, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98+horshift, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95+horshift, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98+horshift, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95+horshift, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98+horshift, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))

# -- Piece 3 --
min_yr = 1995
mx_yr = 2020
Name3 = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name3)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        # df = f[i].pct_change()
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(pd.to_datetime(df).index.values, dtype=float)
        slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
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
rsq1 = round(stats1.loc['rsq', 'CAP'], 4)
rsq2 = round(stats1.loc['rsq', 'Unregulated Groundwater'], 4)
rsq3 = round(stats1.loc['rsq', 'Mixed GW/SW'], 4)
rsq4 = round(stats1.loc['rsq', 'Regulated Groundwater'], 4)
rsq5 = round(stats1.loc['rsq', 'Surface Water'], 4)
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

# ax.plot(xf1, yf1,"-.",color=c_2, lw=1)
ax.plot(xf1, yf2,"-.",color=c_7, lw=1)
ax.plot(xf1, yf3,"-.",color=c_5, lw=1)
ax.plot(xf1, yf4,"-.",color=c_3, lw=1)
# ax.plot(xf1, yf5,"-.",color=c_4, lw=1)

vertshift = -0.3
horshift = 0.6
plt.figtext(0.94+horshift, 0.55 - vertshift, 'Regression for ' +str(min_yr)+' to '+str(mx_yr)+':')
plt.figtext(0.95+horshift, 0.5 - vertshift, 'CAP equation: y = '+str(m1)+'x + '+str(yint1))
plt.figtext(0.98+horshift, 0.45 - vertshift, 'rsq = '+ str(rsq1) + '; p-value = ' + str(pval1))
plt.figtext(0.95+horshift, 0.4 - vertshift, 'Unregulated GW equation: y = '+str(m2)+'x + '+str(yint2))
plt.figtext(0.98+horshift, 0.35 - vertshift, 'rsq = '+ str(rsq2) +'; p-value = ' + str(pval2))
plt.figtext(0.95+horshift, 0.3 - vertshift, 'Mixed SW/GW equation: y = '+str(m3)+'x + '+str(yint3))
plt.figtext(0.98+horshift, 0.25 - vertshift, 'rsq = '+ str(rsq3) +'; p-value = ' + str(pval3))
plt.figtext(0.95+horshift, 0.2 - vertshift, 'Regulated GW equation: y = '+str(m4)+'x + '+str(yint4))
plt.figtext(0.98+horshift, 0.15 - vertshift, 'rsq = '+ str(rsq4) +'; p-value = ' + str(pval4))
plt.figtext(0.95+horshift, 0.1 - vertshift, 'SW equation: y = '+str(m5)+'x + '+str(yint5))
plt.figtext(0.98+horshift, 0.05 - vertshift, 'rsq = '+ str(rsq5) +'; p-value = ' + str(pval5))


# --- Code for Main Plot ---
ds = cat_wl2_SW
minyear=1975
maxyear=2020
min_y = 75
max_y = 300
fsize = 14

# ax.plot(ds['CAP'], label='CAP', color=c_2)
ax.plot(ds['No_CAP'], label='Regulated GW', color=c_3) 
# ax.plot(ds['SW'], label='Surface Water', color=c_4) 
ax.plot(ds['Mix'], label='Mixed SW/GW', color=c_5)
ax.plot(ds['GW'], label='Unregulated GW', color=c_7)  

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
# ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)

ax.minorticks_on()

fig.set_dpi(600.0)

# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(75,300)
# ax.set_title(Name)
ax.set_title('Linear Regression Depth to Water and Access to Surface Water Categories')
ax.legend(
        # loc = [1.065, 0.75]
        )
# plt.savefig(outputpath+'Stats/Water_CAT/'+Name+'_all', bbox_inches = 'tight')
plt.savefig(outputpath+'Stats/Water_CAT/'+Name+'_GW_3pieces', bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'_GW.csv')

# %% For Depth to Water by regulation
ds = cat_wl2_reg
data_type = "Depth to Water"
min_yr = 1975
mx_yr = 2020
betterlabels = ['Regulated','Reservation','Unregulated'] 
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
        # ax.set_ylim(0,max(y))
        # ax.legend()


stats.index = betterlabels
stats1 = stats.transpose()
print(stats1)
#%%
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

fig, ax = plt.subplots(1, 1, figsize = (7,4.5))
ax.plot(xf1, yf1,"-.",color=cap,label='Linear Trendline', lw=1)
ax.plot(xf1, yf2,"-.",color=GWdom, lw=1)

ds = cat_wl2_reg
minyear=1975
maxyear=2020
min_y = 75
max_y = 300
fsize = 12

ax.plot(ds['R'], label='Regulated', color=cap) 
ax.plot(ds['U'], label='Unregulated', color=GWdom) 

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
# # Drought Year Shading
# a = 1988.5
# b = 1990.5
# c = 1995.5
# d = 1996.5
# e = 2001.5
# f = 2003.5
# g = 2005.5
# h = 2007.5
# i = 2011.5
# j = 2014.5
# k = 2017.5
# l= 2018.5
# plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
# plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
# plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

ax.minorticks_on()

fig.set_dpi(600.0)

# ax.set_xlim(min_yr, mx_yr)
ax.set_ylim(75,300)
# ax.set_title(Name)

plt.figtext(0.95, 0.4, 'Regulated equation: y= '+str(m1)+'x + '+str(yint1))
plt.figtext(0.96, 0.35, 'p-value = ' + str(pval1))
plt.figtext(0.95, 0.6, 'Unregulated equation: y= '+str(m2)+'x + '+str(yint2))
plt.figtext(0.96, 0.55, 'p-value = ' + str(pval2))
ax.legend()
plt.savefig(outputpath+'Stats/'+Name, bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/'+Name+'.csv')

# %% ====== Specialized Drought Analysis ======
# Wanting to look at 1) Drawdown 2) Anomaly's 3) Recovery
#   Decided from the drought indices analysis that the cutoff value is -3 for severe droughts

# First read in the drought indice
drought_indices = pd.read_csv('../MergedData/Output_files/Yearly_DroughtIndices.csv')
drought_indices = drought_indices.set_index('In_year')
drought_indices

# %% Drought dictionary
dd = {1:[1989,1990]
        ,2:[1996]
        ,3:[2002,2003]
        ,4:[2006,2007]
        ,5:[2012,2013,2014]
        ,6:[2018]}

print(dd)

#%% Pre-drought
pre_d = {1:[1988]
        ,2:[1995]
        ,3:[2001]
        ,4:[2005]
        ,5:[2011]
        ,6:[2017]}

print(pre_d)

#%% Print the average PDSI and PHDI values

ds = drought_indices.copy()
columns = ds.columns
column_list = ds.columns.tolist()

ds['Status'] = 'Normal-Wet'
# wlanalysis_period

for x,y in dd.items():
        ds.loc[y, 'Status'] = 'Drought '+str(x)


pdsi_avg = ds.groupby(['Status']).mean()
pdsi_avg

#%%
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
# %% Grouped bar chart of PDSI/PHDI Values
name = 'Average PDSI and PHDI Values Per Drought'

yearlabels = ["1989-1990"
                ,'1996'
                ,'2002-2003'
                ,'2006-2007'
                ,'2012-2014'
                ,'2018'
                ,'Normal/Wet Years']

pdsi_avg.index = yearlabels
pdsi_avg = pdsi_avg.transpose()
# del ds['Normal/Wet Years']
pdsi_avg
#%%
group_colors = [blind[5],blind[6],blind[2]
                ,blind[12],blind[11],blind[10]
                ,blind[0] #black
                ]

horlabel = 'Index Value'
fsize = 14

plt.rcParams["figure.dpi"] = 600
pdsi_avg.plot(figsize = (10,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
# plt.ylim([0,400])
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0, fontsize = fsize-2)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(loc=[1.01,0.3],fontsize = fsize)

# plt.savefig(outputpath+name+'_groupedchart', bbox_inches = 'tight')

# %% Figure out which water level database you want
cat_wl2 = cat_wl2_reg.copy() 
# cat_wl2 = cat_wl2_SW.copy()
# cat_wl2 = cat_wl2_georeg.copy()

# cat_wl2 = wdc1_reg.copy()
# cat_wl2 = wdc2_reg.copy()
# cat_wl2 = wdc3_reg.copy()
# cat_wl2 = wdc1_SW.copy()
# cat_wl2 = wdc2_SW.copy()
# cat_wl2 = wdc3_SW.copy()

# Water Analysis period
wlanalysis_period = cat_wl2[cat_wl2.index>=1975]
# wlanalysis_period["UGW"]=wlanalysis_period['GW']
# del wlanalysis_period['GW']
# wlanalysis_period

#%%
# Anomaly's
ds = wlanalysis_period
columns = ds.columns
column_list = ds.columns.tolist()

dtw_anomalys = pd.DataFrame()
for i in column_list:
        dtw_anomalys[i] = wlanalysis_period[i] - wlanalysis_period[i].mean()

dtw_anomalys.head()

# %% Drawdown
ds = wlanalysis_period.copy()
columns = ds.columns
column_list = ds.columns.tolist()

ds['Status'] = 'Normal-Wet'
# wlanalysis_period

for x,y in dd.items():
        ds.loc[y, 'Status'] = 'Drought '+str(x)


drawd_max = ds.groupby(['Status']).max()
drawd_max
#%%
ds = wlanalysis_period.copy()
columns = ds.columns
column_list = ds.columns.tolist()

ds['Status'] = 'Normal-Wet'

for x,y in pre_d.items():
        ds.loc[y, 'pre_d'] = 'Drought '+str(x)

predrought = ds.groupby(['pre_d']).mean()
predrought

# %% Drawdown
drawdown = drawd_max - predrought
drawdown

# %% Checking for normality
# ds = wlanalysis_period
ds = dtw_anomalys
columns = ds.columns
column_list = ds.columns.tolist()

for i in column_list:
 fig, ax = plt.subplots(1,1)
 ax.hist(wlanalysis_period[i], bins=30)
 ax.set_title(i)

# %% If running a shifted correlation analysis,
#    change this to however many # years; 0 is no lag
lag = 0

print('Kendall Correlation coefficient')
for i in column_list:
        # print(' '+i+':')
        print(' '+str(i)+':')
# To normalize the data 
        # df1 = ds[i].pct_change()
        # df2 = drought_indices.PDSI.pct_change()
        df1 = ds[i]
        df2 = drought_indices.PDSI.shift(lag)
        print('  tau = ',round(df1.corr(df2, method='kendall'),3))
        print('  pval = ',round(df1.corr(df2, method=kendall_pval),4))

# %%
print('Spearman Correlation coefficient')
for i in column_list:
        print(' '+str(i)+':')
        # df1 = ds[i].pct_change()
        # df2 = drought_indices.PDSI.pct_change()
        df1 = ds[i]
        df2 = drought_indices.PDSI.shift(lag)
        print('  rho = ',round(df1.corr(df2, method='spearman'),3))
        print('  pval = ',round(df1.corr(df2, method=spearmanr_pval),4))

# %%
print('Pearson Correlation coefficient')
for i in column_list:
        print(' '+str(i)+':')
        # df1 = ds[i].pct_change()
        # df2 = drought_indices.PDSI.pct_change()
        df1 = ds[i]
        df2 = drought_indices.PDSI.shift(lag)
        r = df1.corr(df2, method='pearson')
        print('  rsq = ',round(r*r,3))
        print('  pval = ',round(df1.corr(df2, method=pearsonr_pval),4))


# %% Scatterplot of correlation values
ds = dtw_anomalys
# name = 'Comparing PDSI with Depth to Water Anomalies by Access to SW'
name = 'Comparing PDSI with Depth to Water Anomalies by Regulation'
# del ds['Res']
columns = ds.columns
column_list = ds.columns.tolist()
# betterlabels = ['Receives CAP (Regulated)','GW Dominated (Regulated)','Surface Water Dominated','GW Dominated','Mixed Source'] 
betterlabels = ['Regulated','Unregulated'] 
colors=[cap, GWdom]
# colors=[cap,noCAP, swdom, mixed, GWdom]

fig, ax = plt.subplots(figsize = (7,5))
x = drought_indices['PDSI']
for i,j,k in zip(column_list
                # ,reg_colors
                # , SW_colors
                , colors
                , betterlabels
                ):
        y = ds[i]
        ax.scatter(x,y
                , label=k
                , color=j
                )
        # Trendline: 1=Linear, 2=polynomial
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x),'-'
                , color=j
                # ,label=(k+' Trendline')
                )


ax.set_xlabel('PDSI')
ax.set_ylabel('Depth to Water Anomalies (ft)')
ax.set_title(name,loc='center')
# ax.set_ylim(0,400)
fig.set_dpi(600)
plt.legend(loc = [1.05, 0.40])

plt.savefig(outputpath+name, bbox_inches='tight') 

# %% Grouped bar chart of individual drought anomlies
# cat_wl2 = wdc1_reg.copy() # Deep
# cat_wl2 = wdc2_reg.copy() # Midrange
# cat_wl2 = wdc3_reg.copy() # Shallow
# cat_wl2 = wdc1_SW.copy()
# cat_wl2 = wdc2_SW.copy()
# cat_wl2 = wdc3_SW.copy()
cat_wl2 = cat_wl2_SW
# cat_wl2 = cat_wl2_reg

# name = 'Average DTW Anomalies by Drought Period and Groundwater Regulation'
# name = 'Average DTW Anomalies by Drought Period and Access to SW'

# name = 'Deep Wells'
# name = 'Midrange Wells'
# name = 'Shallow Wells'

betterlabels = ['CAP','Regulated \n Groundwater','Surface \n Water','Unregulated \n Groundwater','Mixed \n GW/SW'] 
# betterlabels = ['GW Regulated','GW Unregulated'] 

yearlabels = ["1989-1990",'1996','2002-2003','2006-2007','2012-2014','2018','Normal/Wet Years']

#%%
# Water Analysis period
wlanalysis_period = cat_wl2[cat_wl2.index>=1975]
# wlanalysis_period["UGW"]=wlanalysis_period['GW']
# del wlanalysis_period['GW']
# wlanalysis_period

# Anomaly's
ds = wlanalysis_period.copy()
columns = ds.columns
column_list = ds.columns.tolist()

dtw_anomalys = pd.DataFrame()
for i in column_list:
        dtw_anomalys[i] = wlanalysis_period[i] - wlanalysis_period[i].mean()

# %%
ds = dtw_anomalys.copy()
# ds = drought_indices

ds['Status'] = 'Normal-Wet'
# wlanalysis_period

for x,y in dd.items():
        ds.loc[y, 'Status'] = 'Drought '+str(x)

ds

ds_indd = ds.groupby(['Status']).mean()
ds_indd.index = yearlabels
ds_indd = ds_indd.transpose()
ds_indd.index = betterlabels
ds_indd

#%%
# group_colors = ['lightsalmon','tomato','orangered','r','brown','indianred','steelblue']

group_colors = [
                blind[5],blind[6],blind[2]
                ,blind[12],blind[11],blind[10]
                ,blind[0] #black
                ]

horlabel = 'DTW Anomaly (ft)'
fsize = 14

ds_indd.plot(figsize = (10,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
# plt.ylim([0,400])
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0, fontsize = fsize-2)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(loc=[1.01,0.3],fontsize = fsize)
# plt.figure(dpi=600)

plt.savefig(outputpath+name+'_anomalies_GWREG_groupedchart', bbox_inches = 'tight')
# plt.savefig(outputpath+name+'_anomalies_SWAccess_groupedchart', bbox_inches = 'tight')

#%% Drawdown quick analysis
# cat_wl2 = cat_wl2_reg.copy() 
cat_wl2 = cat_wl2_SW.copy()
# cat_wl2 = cat_wl2_georeg.copy()

# cat_wl2 = wdc1_reg.copy()
# cat_wl2 = wdc2_reg.copy()
# cat_wl2 = wdc3_reg.copy()
# cat_wl2 = wdc1_SW.copy()
# cat_wl2 = wdc2_SW.copy()
# cat_wl2 = wdc3_SW.copy()

betterlabels = ['CAP','Regulated \n Groundwater','Surface \n Water','Unregulated \n Groundwater','Mixed \n GW/SW'] 
# betterlabels = ['GW Regulated','GW Unregulated'] 

# ---
wlanalysis_period = cat_wl2[cat_wl2.index>=1975]

ds = wlanalysis_period.copy()
columns = ds.columns
column_list = ds.columns.tolist()
ds['Status'] = 'Normal-Wet'
for x,y in dd.items():
        ds.loc[y, 'Status'] = 'Drought '+str(x)

for x,y in pre_d.items():
        ds.loc[y, 'pre_d'] = 'Drought '+str(x)
# ds

drawd_max = ds.groupby(['Status']).max()
predrought = ds.groupby(['pre_d']).mean()

drawdown = drawd_max - predrought
drawdown

#%% Grouped Bar chart for drawdown (ft)
# name = 'Max Drawdown by Drought Period and Groundwater Regulation'
name = 'Max Drawdown by Drought Period and Access to SW'

yearlabels = ["1989-1990",'1996','2002-2003','2006-2007','2012-2014','2018','Normal/Wet Years']

drawdown.index = yearlabels
drawdown = drawdown.transpose()
drawdown.index = betterlabels
del drawdown['Normal/Wet Years']
drawdown

#%% 
# group_colors = ['lightsalmon','tomato','orangered','r','brown','indianred','steelblue']

group_colors = [blind[5],blind[6],blind[2]
                ,blind[12],blind[11],blind[10]
                ,blind[0] #black
                ]

horlabel = 'Drawdown (ft)'
fsize = 14

plt.rcParams["figure.dpi"] = 600
drawdown.plot(figsize = (10,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
# plt.ylim([0,400])
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0, fontsize = fsize-2)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(loc=[1.01,0.3],fontsize = fsize)
# plt.set_dpi(600)

# plt.savefig(outputpath+name+'_GWREG_groupedchart', bbox_inches = 'tight')
plt.savefig(outputpath+name+'_anomalies_SWAccess_groupedchart', bbox_inches = 'tight')

# %% --- Recovery ---
cat_wl2 = cat_wl2_reg.copy() 
# cat_wl2 = cat_wl2_SW.copy()
# cat_wl2 = cat_wl2_georeg.copy()

# cat_wl2 = wdc1_reg.copy()
# cat_wl2 = wdc2_reg.copy()
# cat_wl2 = wdc3_reg.copy()
# cat_wl2 = wdc1_SW.copy()
# cat_wl2 = wdc2_SW.copy()
# cat_wl2 = wdc3_SW.copy()

# betterlabels = ['CAP','Regulated \n Groundwater','Surface \n Water','Unregulated \n Groundwater','Mixed \n GW/SW'] 
betterlabels = ['GW Regulated','GW Unregulated'] 

# ---
wlanalysis_period = cat_wl2[cat_wl2.index>=1975]

ds = wlanalysis_period.copy()
columns = ds.columns
column_list = ds.columns.tolist()
ds['Status'] = 'Normal-Wet'
for x,y in dd.items():
        ds.loc[y, 'Status'] = 'Drought '+str(x)

for x,y in pre_d.items():
        ds.loc[y, 'pre_d'] = 'Drought '+str(x)
ds


# %% making a list of droughts for looping
droughts = ds['Status'].unique()
droughtslist = droughts.tolist()
del droughtslist[0]
droughtslist

#%% Year when drought is at it's minimum (start_val)
df = ds.copy()
start_val = pd.DataFrame(index=droughtslist,columns=column_list)
for i in droughtslist:
        lol = df[(df['Status']==i)] # This narrows to the drought of interest
        for n in column_list:
                thing = lol[lol[n]==lol[n].max()].index.tolist() # This pulls out the year
                start_val.loc[i,n] = thing[0]
        # df
start_val = start_val.astype(float) # This converts the object to float for calculations


#%% Year when drought recovered (end_val)
df = ds.copy()
end_val = pd.DataFrame(index=droughtslist,columns=column_list)
for i in droughtslist:
        #this bit will grab the max year
        lol = df[(df['Status']==i)] # This narrows to the drought of interest for the max year
        lol2 = df[(df['pre_d']==i)] # This makes a dataframe of predrought values
        for n in column_list:
                thing = lol[lol[n]>=lol[n].max()].index.tolist() # This pulls out the year
                year = thing[0]
                newdata = df[df.index>=year] # now we have eliminated the prior years
                pre_dval = lol2[n].mean()
                rec_yeardf = newdata[newdata[n]<=pre_dval]
                listy = rec_yeardf.index.tolist()
                print(listy)
                if len(listy)==0:
                    print ("no recovery")
                    
                else:
                  print ("yay recovery")
                  end_val.loc[i,n] = listy[0]
        # df
end_val = end_val.astype(float)
end_val

# %%
recoverytime = end_val - start_val
recoverytime

# name = 'Recovery Time by Drought Period and Groundwater Regulation'
name = ' by Drought Period and Access to SW'

yearlabels = ["1989-1990",'1996','2002-2003','2006-2007','2012-2014','2018']

recoverytime.index = yearlabels
recoverytime = recoverytime.transpose()
recoverytime.index = betterlabels
# del recoverytime['Normal/Wet Years']
recoverytime

# %%
recoverytime = recoverytime.transpose()

#%% 
# group_colors = ['lightsalmon','tomato','orangered','r','brown','indianred','steelblue']

group_colors = [blind[5],blind[6],blind[2]
                ,blind[12],blind[11],blind[10]
                ,blind[0] #black
                ]

horlabel = 'Time (years)'
fsize = 14

plt.rcParams["figure.dpi"] = 600
recoverytime.plot(figsize = (10,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        # color = reg_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
# plt.ylim([0,400])
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0, fontsize = fsize-2)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(loc=[1.01,0.3],fontsize = fsize)
# plt.set_dpi(600)

plt.savefig(outputpath+name+'_groupedchart', bbox_inches = 'tight')
# plt.savefig(outputpath+name+'_groupedchart', bbox_inches = 'tight')

# plt.savefig(outputpath+name+'_anomalies_SWAccess_groupedchart', bbox_inches = 'tight')


# %% Now to do a box plot or bar plot
# Assign severe values based on years
ds = dtw_anomalys

ds['Status'] = 'Other'
# wlanalysis_period

Drought_years = [1989,1990,1996,2002,2003,2006,2007,2012,2014,2018]


for i in Drought_years:
        ds.loc[i, 'Status'] = 'Severe'

ds

# Severe drought dataframe and normal
severe = ds[ds['Status']=='Severe']
severe

other = ds[ds['Status']=='Other']
other

del severe['Status']
del other['Status']

# %% Grouped bar chart for regulation and SW (just gotta turn on and off different things)
# betterlabels = ['CAP','Regulated \n Groundwater','Surface \n Water','Unregulated \n Groundwater','Mixed \n GW/SW'] 
betterlabels = ['GW Regulated','GW Unregulated'] 

# name = 'Average DTW Anomalys by Drought and Groundwater Regulation'
# name = 'Average DTW Anomalys by Drought and Access to SW'

# name = 'Average Depth to water by Drought and Groundwater Regulation'
# name = 'Average Depth to water by Drought and Access to SW'
# name = 'Deep Wells'
# name = 'Midrange Wells'
# name = 'Shallow Wells'

ds = severe.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
for i in labels:
        df = ds[i]
        # print(df)
        y=np.array(df.values, dtype=float)
        print(y)
        ds1 = ds1.append({'Severe': np.mean(y)},
                              ignore_index=True)
ds1

dft1 = ds1.copy()
dft1.index = betterlabels
dft1 = dft1.transpose()
dft1

ds = other.copy()
# ds = dens_wdc2.copy()
columns = ds.columns
labels = ds.columns.tolist()

ds1 = pd.DataFrame()
for i in labels:
        df = ds[i]
        # print(df)
        y=np.array(df.values, dtype=float)
        print(y)
        ds1 = ds1.append({'Normal-Wet': np.mean(y)},
                              ignore_index=True)
dft2 = ds1.copy()
dft2

dft2.index = betterlabels
dft2 = dft2.transpose()

df_test = dft1.append([dft2])
df_test = df_test.transpose()

# group_colors = ['lightsteelblue','cornflowerblue','darkblue']
# group_colors = reg_colors
group_colors = [c_3,'lightsteelblue']

horlabel = 'DTW Anomaly (ft)'
fsize = 14

df_test.plot(figsize = (7,7),
        kind='bar',
        stacked=False,
        # title=name,
        color = group_colors,
        zorder = 2,
        width = 0.85,
        fontsize = fsize
        )
plt.title(name, fontsize = (fsize+2))
# plt.ylim([0,400])
plt.ylabel(horlabel, fontsize = fsize)
plt.xticks(rotation=0, fontsize = fsize-2)
plt.grid(axis='y', linewidth=0.5, zorder=0)
plt.legend(fontsize = fsize)

plt.savefig(outputpath+name+'_anomalies_GWREG_groupedchart', bbox_inches = 'tight')
# plt.savefig(outputpath+name+'_anomalies_SWAccess_groupedchart', bbox_inches = 'tight')


# %% Plotting with PDSI against time to see if there's a relationship with Access to SW
ds = cat_wl2_SW
minyear=1975
maxyear=2020
lag = -4
# name = "Average DTW and PDSI from " + str(minyear) + " to " + str(maxyear) + ' lagged by ' + str(lag)
name = "Average DTW and PDSI from " + str(minyear) + " to " + str(maxyear)

min_y = 0
max_y = 300
fsize = 14

fig, ax = plt.subplots(figsize = (9,6))
# ax.plot(ds['R'].shift(lag), label='GW Regulated', color=c_2) 
# ax.plot(ds['U'].shift(lag), label='GW Unregulated', color=c_7)

ax.plot(ds['CAP'], label='CAP', color=c_2)
ax.plot(ds['No_CAP'], label='Regulated Groundwater', color=c_3)
ax.plot(ds['SW'], label='Surface Water', color=c_4)
ax.plot(ds['GW'], label='Unregulated Groundwater', color=c_7)
ax.plot(ds['Mix'], label='Mixed SW/GW', color=c_5)
# colors = [c_2,c_3,c_4,c_7,c_5]

# Secondary Axis
ax2 = ax.twinx()
ax2.set_ylabel('PDSI')
ax2.set_ylim(-7, 10)
ax2.plot(drought_indices['PDSI'], '-.',label='PDSI', color='grey', lw = 3, zorder=0) 

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
# ax.grid(True)
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)

ax.set_title(name, fontsize=20)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Depth to Water (ft)',fontsize=fsize)

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
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Severe Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)

ax.legend(loc = [1.09, 0.45], fontsize = fsize)
ax2.legend(loc = [1.09, 0.30], fontsize = fsize)

fig.set_dpi(600.0)

# plt.savefig(outputpath+name+'_byregulation', bbox_inches='tight')
# plt.savefig(outputpath+name+'_byregulation_Drought', bbox_inches='tight')
# plt.savefig(outputpath+name+'_GWReg_Drought', bbox_inches='tight')
# plt.savefig(outputpath+name+'_GW_Drought', bbox_inches='tight')
# plt.savefig(outputpath+name+'_SW_Drought', bbox_inches='tight')
# plt.savefig(outputpath+name+'_AllAccess_Drought', bbox_inches='tight') 



# %%
# Boxplot Stuff
df = severe
df2 = other
name = 'Severe'
# labels = df.columns.tolist()
# betterlabels = ['CAP','Regulated Groundwater','Surface Water','Unregulated Groundwater','Mixed GW/SW'] 
betterlabels = ['GW Regulated','GW Unregulated'] 

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

bplot = ax1.boxplot(df,
                     vert=True,  
                     patch_artist=True,  
                     labels=betterlabels
                     )

colors = reg_colors
# colors = SW_colors
# colors = [c_2,c_3,c_4,c_7,c_5]


for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_title(name)
plt.xticks(rotation=30)
ax1.set_ylabel('Depth to Water (ft)')
ax1.grid(visible=True)
fig.set_dpi(600.0)
ax1.set_ylim(0,300)

# plt.savefig(outputpath+'Stats/Water_CAT/'+name+"Reverse_axes", bbox_inches = 'tight')
# plt.savefig(outputpath+'Stats/Regulation/'+name+"Reverse_axes", bbox_inches = 'tight')


# %%
name = 'Normal-Wet'
# labels = df.columns.tolist()
# betterlabels = ['CAP','Regulated Groundwater','Surface Water','Unregulated Groundwater','Mixed GW/SW'] 

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

bplot = ax1.boxplot(df2,
                     vert=True,  
                     patch_artist=True,  
                     labels=betterlabels
                     )

colors = reg_colors
# colors = SW_colors
# colors = [c_2,c_3,c_4,c_7,c_5]


for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_title(name)
plt.xticks(rotation=30)
ax1.set_ylabel('Depth to Water (ft)')
ax1.grid(visible=True)
fig.set_dpi(600.0)
ax1.set_ylim(0,300)

# plt.savefig(outputpath+'Stats/Regulation/'+name, bbox_inches = 'tight')
# plt.savefig(outputpath+'Stats/Water_CAT/'+name+'Reverse_axes', bbox_inches = 'tight')

# %% Running a regression on PDSI and access to SW because yolo
ds = cat_wl2
data_type = "Depth to Water and PDSI"
min_yr = 1975
mx_yr = 2020
betterlabels = ['CAP','Unregulated Groundwater','Mixed GW/SW','Regulated Groundwater','Surface Water'] 
Name = str(min_yr) + " to " + str(mx_yr) + " Linear Regression for " + data_type
print(Name)

f = ds[(ds.index >= min_yr) & (ds.index <= mx_yr)]
columns = ds.columns
column_list = ds.columns.tolist()
# -- For Multiple years --
# Name = "Linear Regression during Wet and Normal years for " + data_type
# wetyrs = [2005, 2008, 2009, 2010, 2016, 2017, 2019]
# dryyrs = [2002, 2003, 2004, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2018]
# dryyrs = [1975,1976,1977
#           ,1981,1989,1990
#           ,1996,1997,
#           1999,2000,2001,2002,2003,2004
#           ,2006,2007,2008,2009
#           ,2011, 2012, 2013, 2014, 2015, 2016,2017,2018]
# wetyrs = [1978,1979,1980,1982,1983,1984,1984,1986,1987,1988
#           , 1991,1992,1993,1994,1995,
#           1998,2005,2010,2019]

#f = ds[(ds.index == wetyrs)]

# f = pd.DataFrame()
# for i in wetyrs:
#         wut = ds[(ds.index == i)]
#         f = f.append(wut)
# print(f)
columns = ds.columns
column_list = ds.columns.tolist()
# ------------------------

stats = pd.DataFrame()
for i in column_list:
        df = f[i]
        #print(df)
        y=np.array(df.values, dtype=float)
        x=np.array(drought_indices['PDSI'].values, dtype=float)
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
        # ax.set_ylim(0,max(y))
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
ax.set_ylim(75,300)
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
# plt.savefig(outputpath+'Stats/Water_CAT/'+Name, bbox_inches = 'tight')
# stats1.to_csv(outputpath+'Stats/Water_CAT/'+Name+'.csv')

# %%
# Subtracting off the trend for access to SW
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
        # ax.set_ylim(0,max(y))
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
# df2.plot(ax=ax,marker='o', ls='', label=betterlabels)

# ax.set_xlim(min_yr, mx_yr)
# ax.set_ylim(75,300)
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
# ax.set_ylim(min_y,max_y)
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
