# === GRACE Spatial Analysis Script ===
# written by Danielle Tadych
# The purpose of this script is to analyze GRACE Data for Arizona by points and shapes
#  - Importing packages: Line 6
#  - Reading in files: Line 37
#  - EASYMORE Remapping using a shapefile to computer zonal statistics: Line 56
#     *Note: in order for this package to work
#               > the projections need to be in espg4326, exported, re-read in
#               > the time variable for nc needs to be datetime
#               > Value error -> run the "fix geometries" tool in qgis
#  - Plotting a single grid cell based on lat/lon: Line 336
#  - Calculating the average based off a mask (not weighted): Line 510
# %%
from cProfile import label
from calendar import calendar
from importlib.resources import path
from itertools import count
import os
from pydoc import cli
from typing import Mapping
#import affine
from geopandas.tools.sjoin import sjoin
import matplotlib
from matplotlib.cbook import report_memory
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime as dt
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box, geo
import geopandas as gp
import xarray as xr
import rioxarray as rxr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4
import rasterio
from rasterstats import zonal_stats
#import rasterstats as rstats
#from xrspatial import zonal_stats
import easymore
import glob
# %% Read in the file
filename = 'CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc'
datapath = '../../GRACE'
print(datapath)
outputpath = '../MergedData/Output_files/'

grace_dataset = xr.open_dataset(datapath+'/'+filename)

grace_dataset

# %% Read in the mask shapefile
filename = "Final_Georegions.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles/Final_Georegions/', filename)
georeg = gp.read_file(filepath)

# %%
filename = "AZ_counties.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles', filename)
counties = gp.read_file(filepath)

# %% Read in the mask shapefile
filename = "Ag_NonAG.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles/Final_Georegions/', filename)
agreg = gp.read_file(filepath)

# %% Look at that sweet sweet data
metadata = grace_dataset.attrs
metadata

# %% View first 5 values
grace_dataset["lwe_thickness"]["lat"].values[:5]
print("The min and max latitude values in the data is:", 
      grace_dataset["lwe_thickness"]["lat"].values.min(), 
      grace_dataset["lwe_thickness"]["lat"].values.max())
print("The min and max longitude values in the data is:", 
      grace_dataset["lwe_thickness"]["lon"].values.min(), 
      grace_dataset["lwe_thickness"]["lon"].values.max())

print("The earliest date in the data is:", 
    grace_dataset["lwe_thickness"]["time"].values.min())
print("The latest date in the data is:", 
    grace_dataset["lwe_thickness"]["time"].values.max())

# %%
print("Number of Datapoints")
grace_dataset["lwe_thickness"]['time'].values.shape   

# %% Slicing data to get variables
lat = grace_dataset.variables['lat'][:]
lon = grace_dataset.variables['lon'][:]
time = grace_dataset.variables['time'][:]
lwe = grace_dataset['lwe_thickness']
print(lwe)

# %% Now I need to assign a coordinate system to lwe
lwe.coords['lon'] = (lwe.coords['lon'] + 180) % 360 - 180
#print(lwe['lon'])

lwe2 = lwe.sortby(lwe.lon)
#print(lwe2['lon'])

lwe2 = lwe2.rio.set_spatial_dims('lon', 'lat')
lwe2.rio.crs

lwe2 = lwe2.rio.set_crs("epsg:4269")
lwe2.rio.crs

# %% Convert time to datetime format
# https://stackoverflow.com/questions/38691545/python-convert-days-since-1990-to-datetime-object
# http://unidata.github.io/netcdf4-python/#netCDF4.num2date
# time = ncfile.variables['time'] # do not cast to numpy array yet 
# time_convert = netCDF4.num2date(time[:], time.units, time.calendar)

time = grace_dataset.variables['time'] # do not cast to numpy array yet 
#time

time_convert = netCDF4.num2date(time[:], "days since 2002-01-01T00:00:00Z", calendar='standard')
#time_convert

lwe2['time'] = time_convert
#lwe2

# Converting to the proper datetime format for statistical analyses
#nor_xr is  dataarray (var) name
datetimeindex = lwe2.indexes['time'].to_datetimeindex()

# %% 
lwe2['time'] = datetimeindex
lwe2
# %% Export to raster for graphing
#lwe2.rio.to_raster(r"testGRACE_time.tif")
#lwe2.rio.to_raster(r"testGRACE_time.nc")
# This wrote a tif!!!
# %% Plot the new dataset
lwe2[2,:,:].plot()

# %% Plot the new geotif
da = xr.open_rasterio("testGRACE_time.tif")
#transform = cartopy.Affine.from_gdal(*da.attrs["transform"]) # this is important to retain the geographic attributes from the file
#da = lwe2
# %%
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.imshow(da.variable.data[1])
plt.show()
# %%
lon_min = da.y.min
lon_max = da.y.max
lat_min = da.x.min
lat_max = da.x.max
print(lon_min)
# %%
crs=ccrs.PlateCarree()
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection=crs)
ax.coastlines(resolution='10m', alpha=0.1)
ax.contourf(da.x, da.y, da.variable.data[201], cmap='viridis')
ax.set_extent([lon_min, lon_max, lat_min, lat_max])
# Grid and Labels
gl = ax.gridlines(crs=crs, draw_labels=True, alpha=0.5)
gl.xlabels_top = None
gl.ylabels_right = None
xgrid = np.arange(lon_min-0.5, lon_max+0.5, 1.)
ygrid = np.arange(lat_min, lat_max+1, 1.)
plt.show()

# %% Trying to think of how to lump these attribute tables


# %% ---- Remapping using EASYMORE Package ----
# Prepping the data
# reprojecting coordinate system
#reproject = counties.to_crs(epsg=4326)
# reproject = georeg.to_crs(epsg=4326)
reproject = agreg.to_crs(epsg=4326)
reproject.crs
# %%
# reproject.to_file("georeg_reproject.shp")
reproject.to_file("agreg_reproject.shp")
# %%
reproject.plot()

# %% Now fixing GRACE to be datetime.  Can skip to line 186 if already ran
grace2 = grace_dataset
grace2
# %%
grace2['time'] = datetimeindex
grace2
# %% Write the full nc file
fn = "testGRACE_time.nc"
grace2.to_netcdf(fn)
# %% Now remapping following this tutorial
# https://github.com/ShervanGharari/EASYMORE/blob/main/examples/Chapter1_E1.ipynb
# # loading EASYMORE
from easymore.easymore import easymore

# initializing EASYMORE object
esmr = easymore()

# specifying EASYMORE objects
# name of the case; the temporary, remapping and remapped file names include case name
esmr.case_name                = 'easymore_GRACE_AZ_irrAg'              
# temporary path that the EASYMORE generated GIS files and remapped file will be saved
esmr.temp_dir                 = '../temporary/'

# name of target shapefile that the source netcdf files should be remapped to
# For this test, goign to use counties
# esmr.target_shp               = '../data/target_shapefiles/South_Saskatchewan_MedicineHat.shp'
#remap_shapefile               = '/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles/AZ_counties.shp'
# It needs me to reproject my shapefile to WGS84 (epsg:4326)
#       Note: It said please
#counties  = counties.to_crs(epsg=4326)
#counties.crs

# esmr.target_shp = remap_shapefile.to_crs(espg:4326)
#esmr.target_shp = '/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles/AZ_counties.shp'
#esmr.target_shp = '/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Output_files/Georegions_AGU.shp'
#esmr.target_shp = 'Georegions_reproject.shp'
# esmr.target_shp = 'georeg_reproject_fixed.shp'
esmr.target_shp = 'Ag_NonAG_reproject_fixedgeom.shp'
#esmr.target_shp = 'counties_reproject.shp'
# name of netCDF file(s); multiple files can be specified with *
# esmr.source_nc                = '../data/Source_nc_ERA5/ERA5_NA_*.nc'
# esmr.source_nc                = lwe2
#esmr.source_nc                = '../../GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc'
esmr.source_nc                = 'testGRACE_time.nc'

# name of variables from source netCDF file(s) to be remapped
esmr.var_names                = ["lwe_thickness"]
# rename the variables from source netCDF file(s) in the remapped files;
# it will be the same as source if not provided
esmr.var_names_remapped       = ["lwe_thickness"]
# name of variable longitude in source netCDF files
esmr.var_lon                  = 'lon'
# name of variable latitude in source netCDF files
esmr.var_lat                  = 'lat'
# name of variable time in source netCDF file; should be always time
esmr.var_time                 = 'time'
# location where the remapped netCDF file will be saved
esmr.output_dir               = outputpath
# format of the variables to be saved in remapped files,
# if one format provided it will be expanded to other variables
esmr.format_list              = ['f4']
# fill values of the variables to be saved in remapped files,
# if one value provided it will be expanded to other variables
esmr.fill_value_list          = ['-9999.00']
# if required that the remapped values to be saved as csv as well
esmr.save_csv                 = True
esmr.complevel                 =  9
# if uncommented EASYMORE will use this and skip GIS tasks
#esmr.remap_csv                = '../temporary/ERA5_Medicine_Hat_remapping.csv'
# %%
# execute EASYMORE
esmr.nc_remapper()

# %%
# visualize the remapped netCDF for the first file, first time step
# target nc file
nc_names = sorted(glob.glob (esmr.output_dir + esmr.case_name + '*.nc'))
ds       = xr.open_dataset(nc_names[0]) # the first netcdf file
values   = ds.lwe_thickness [0,:] # the first time frame of the first 
IDs      = ds.ID [:] # get the ID
# create a data frame for the model simulation
df = pd.DataFrame()
df ['value'] = values
df ['ID_t']    = IDs  # .astype(int)
df = df.sort_values (by = 'ID_t')
# load the shape file target that is generated by EASYMORE (with consistent IDs)
shp_target = gp.read_file(esmr.temp_dir+ esmr.case_name + '_target_shapefile.shp') # load the target shapefile
shp_target ['ID_t'] = shp_target ['ID_t'].astype(float)
shp_target = shp_target.sort_values(by='ID_t')# sort on values
shp_target = pd.merge_asof(shp_target, df, on='ID_t', direction='nearest')
shp_target = shp_target.set_geometry('geometry') #bring back the geometry filed; pd to gpd
# plotting
f, axes = plt.subplots(1,1,figsize=(15,15))
#ds_source.airtemp[0,:,:].plot( ax = axes)
#ds_source.airtemp[0,:,:].plot( ax = axes, alpha = 1, add_colorbar=False)
shp_target.plot(column= 'value', edgecolor='k',linewidth = 1, ax = axes , legend=True)
#plt.savefig('../fig/Example1_B.png')

# %% Now, read in the remapped csv
# filename = 'easymore_GRACE_georeg_remapped_lwe_thickness__2002-04-18-00-00-00.csv'
filename = 'easymore_GRACE_AZ_irrAg_remapped_lwe_thickness__2002-04-18-00-00-00.csv'
filepath = os.path.join(outputpath, filename)
grace_remapped = pd.read_csv(filepath)
grace_remapped.head()
# %% Gotta fix those headers
# ID_key = shp_target[['GEO_Region', 'ID_t']]
ID_key = shp_target[['GRACE_Cat', 'ID_t']]
ID_key

# %%
ID_key['ID'] = 'ID_' + ID_key['ID_t'].astype(str)
ID_key

#%%
grace_remapped = grace_remapped.set_index('time')

# %%
del grace_remapped['Unnamed: 0'] # tbh not really sure why this column is here but gotta delete it

# %%
grace_remapped
# %%
georeg_list = ID_key['GRACE_Cat'].values.tolist()
georeg_list

# %%
grace_remapped.columns = georeg_list
grace_remapped

# %% Fixing the time element
grace_remapped.index = pd.to_datetime(grace_remapped.index)
grace_remapped
# %%
grace_remapped.plot()

# %%
grace_yearly = grace_remapped
grace_yearly['year'] = pd.DatetimeIndex(grace_yearly.index).year
grace_yearly

# %%
grace_yearly = grace_yearly.reset_index()
# %%
#grace_yearly = grace_yearly.set_index('time')
grace_yearly

# %%
grace_yearlyavg = pd.pivot_table(grace_yearly, index=["year"], dropna=False, aggfunc=np.mean)
grace_yearlyavg
#%%
grace_yearlyavg = grace_yearlyavg.reset_index()
grace_yearlyavg

# %%
grace_yearlyavg['year'] = pd.to_numeric(grace_yearlyavg['year'])
grace_yearlyavg['year'] = grace_yearlyavg['year'].astype(int)
grace_yearlyavg.info()
#%%
grace_yearlyavg = grace_yearlyavg.set_index('year')
grace_yearlyavg

# %%
grace_yearlyavg.plot()

# %% Now let's do this for months
# %%
grace_monthly = grace_remapped
grace_monthly['month'] = pd.DatetimeIndex(grace_monthly.index).month
grace_monthly

# %%
grace_monthly = grace_monthly.reset_index()

# %%
grace_monthlyavg = pd.pivot_table(grace_monthly, index=["month"], dropna=False, aggfunc=np.mean)
grace_monthlyavg

del grace_monthlyavg['year']
#%%
grace_monthlyavg = grace_monthlyavg.reset_index()
grace_monthlyavg

# %%
grace_monthlyavg['month'] = pd.to_numeric(grace_monthlyavg['month'])
grace_monthlyavg['month'] = grace_monthlyavg['month'].astype(int)
grace_monthlyavg = grace_monthlyavg.set_index('month')
grace_monthlyavg

# %%
grace_monthlyavg.plot()
# %% --- Plotting the Remapped data ---
# Creating colors
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

# %% For plotting the Monthly Average
ds = grace_monthlyavg
name = "Seasonal Average (Interannual) Change from 2004-2009 Baseline"
ylabel = "Liquid Water Equivalent (cm)"
minyear=2002
maxyear=2020
min_y = -15
max_y = 7

# Plot all of them
fig, ax = plt.subplots(2,2,figsize=(24,12))
#fig.tight_layout()
fig.suptitle(name, fontsize = 20, y = 0.91)
fig.supylabel(ylabel, fontsize = 20, x = 0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0,0].plot(ds['Regulated with CAP'], label='Regulated with CAP', color=c_2) 
ax[0,0].plot(ds['Regulated without CAP'], label='Regulated without CAP', color=c_3) 
ax[1,1].plot(ds['Lower Colorado River - SW Dominated'], color=c_4, label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds['Upper Colorado River - Mixed'], color=c_5, label='Upper Colorado River - Mixed')
ax[0,1].plot(ds['Norh - Mixed'], color=c_10, label='North - Mixed')
ax[0,1].plot(ds['Central - Mixed'], color=c_11, label='Central - Mixed')
ax[1,0].plot(ds['Northwest - GW Dominated'], color=c_7, label='Northwest - GW Dominated')
ax[1,0].plot(ds['Northeast - GW Dominated'], color=c_9, label='Northeast - GW Dominated')
ax[1,0].plot(ds['South central - GW Dominated'], color=c_8, label='South central - GW Dominated')
ax[1,0].plot(ds['Southeast - GW Dominated'], color=c_6, label='Southeast - GW Dominated')
#ax[0,0].set_xlim(minyear,maxyear)
#ax[0,1].set_xlim(minyear,maxyear)
#ax[1,0].set_xlim(minyear,maxyear)
#ax[1,1].set_xlim(minyear,maxyear)
#ax[0,0].set_ylim(min_y,max_y)
#ax[0,1].set_ylim(min_y,max_y)
#ax[1,0].set_ylim(min_y,max_y)
#ax[1,1].set_ylim(min_y,max_y)
ax[0,0].grid(True)
ax[0,1].grid(True)
ax[1,0].grid(True)
ax[1,1].grid(True)
#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
ax[0,0].legend(loc = [0.55, 0.60], fontsize=14)
ax[0,1].legend(loc = [0.55, 0.6], fontsize=14)
ax[1,0].legend(loc = [0.55, 0.6], fontsize=14)
ax[1,1].legend(loc = [0.45, 0.75], fontsize=14)

plt.savefig(outputpath+name)

#%% For Plotting the Yearly average of Georegions
ds = grace_yearlyavg
name = "Yearly Average (Intra-annual) Change from 2004-2009 Baseline"
minyear=2002
maxyear=2020
min_y = -15
max_y = 7
ylabel = "Liquid Water Equivalent (cm)"

# Plot all of them
fig, ax = plt.subplots(2,2,figsize=(24,12))
#fig.tight_layout()
fig.suptitle(name, fontsize = 20, y = 0.91)
fig.supylabel(ylabel, fontsize = 20, x = 0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color=c_1)
ax[0,0].plot(ds['Regulated with CAP'], label='Regulated with CAP', color=c_2) 
ax[0,0].plot(ds['Regulated without CAP'], label='Regulated without CAP', color=c_3) 
ax[1,1].plot(ds['Lower Colorado River - SW Dominated'], color=c_4, label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds['Upper Colorado River - Mixed'], color=c_5, label='Upper Colorado River - Mixed')
ax[0,1].plot(ds['Norh - Mixed'], color=c_10, label='North - Mixed')
ax[0,1].plot(ds['Central - Mixed'], color=c_11, label='Central - Mixed')
ax[1,0].plot(ds['Northwest - GW Dominated'], color=c_7, label='Northwest - GW Dominated')
ax[1,0].plot(ds['Northeast - GW Dominated'], color=c_9, label='Northeast - GW Dominated')
ax[1,0].plot(ds['South central - GW Dominated'], color=c_8, label='South central - GW Dominated')
ax[1,0].plot(ds['Southeast - GW Dominated'], color=c_6, label='Southeast - GW Dominated')
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
#ax[0,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='bottom', fontsize=14)
ax[0,0].legend(loc = [0.1, 0.20], fontsize=14)
ax[0,1].legend(loc = [0.1, 0.05], fontsize=14)
ax[1,0].legend(loc = [0.1, 0.05], fontsize=14)
ax[1,1].legend(loc = [0.1, 0.20], fontsize=14)
plt.savefig(outputpath+name)

# %% For plotting all the data
ds = grace_remapped
name = "All GRACE Data (not averaged)"
minyear=2002
maxyear=2020
min_y = -15
max_y = 7

# Plot all of them
fig, ax = plt.subplots(2,2,figsize=(24,12))
#fig.tight_layout()
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0,0].plot(ds['Regulated with CAP'], label='Regulated with CAP', color=c_2) 
ax[0,0].plot(ds['Regulated without CAP'], label='Regulated without CAP', color=c_3) 
ax[1,1].plot(ds['Lower Colorado River - SW Dominated'], color=c_4, label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds['Upper Colorado River - Mixed'], color=c_5, label='Upper Colorado River - Mixed')
ax[0,1].plot(ds['Norh - Mixed'], color=c_10, label='North - Mixed')
ax[0,1].plot(ds['Central - Mixed'], color=c_11, label='Central - Mixed')
ax[1,0].plot(ds['Northwest - GW Dominated'], color=c_7, label='Northwest - GW Dominated')
ax[1,0].plot(ds['Northeast - GW Dominated'], color=c_9, label='Northeast - GW Dominated')
ax[1,0].plot(ds['South central - GW Dominated'], color=c_8, label='South central - GW Dominated')
ax[1,0].plot(ds['Southeast - GW Dominated'], color=c_6, label='Southeast - GW Dominated')
#ax[0,0].set_xlim(minyear,maxyear)
#ax[0,1].set_xlim(minyear,maxyear)
#ax[1,0].set_xlim(minyear,maxyear)
#ax[1,1].set_xlim(minyear,maxyear)
#ax[0,0].set_ylim(min_y,max_y)
#ax[0,1].set_ylim(min_y,max_y)
#ax[1,0].set_ylim(min_y,max_y)
#ax[1,1].set_ylim(min_y,max_y)
ax[0,0].grid(True)
ax[0,1].grid(True)
ax[1,0].grid(True)
ax[1,1].grid(True)
#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[0,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='bottom', fontsize=14)
ax[0,0].legend(loc = [0.1, 0.20], fontsize=14)
ax[0,1].legend(loc = [0.1, 0.05], fontsize=14)
ax[1,0].legend(loc = [0.1, 0.05], fontsize=14)
ax[1,1].legend(loc = [0.1, 0.20], fontsize=14)
plt.savefig(outputpath+name)

# %% Write a .csv for now for graphing later
grace_remapped.to_csv('../MergedData/Output_files/grace_remapped.csv')
grace_yearlyavg.to_csv('../MergedData/Output_files/grace_remapped_yearly.csv')
#%%
grace_monthlyavg.to_csv('../MergedData/Output_files/grace_remapped_monthly.csv')

# %% ---- Plotting Points ----
key = 400
#longitude = grace_dataset["lwe_thickness"]["lon"].values[key]
#latitude = grace_dataset["lwe_thickness"]["lat"].values[key]
longitude = -110.911
latitude = 32.253
print("Long, Lat values:", longitude, latitude)

# Lat an longitude of Tucson
# 32.253460, -110.911789.

# Create a spatial map of your selected location with cartopy

# Set the spatial extent to cover the CONUS (Continental United States)
extent = [-120, -70, 24, 50.5]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])

# Create your figure and axis object
# Albers equal area is a common CRS used to make maps of the United States
f, ax = plt.subplots(figsize=(12, 6),
                     subplot_kw={'projection': ccrs.AlbersEqualArea(central_lon, central_lat)})
ax.coastlines()
# Plot the selected location
ax.plot(longitude, latitude, 
        '*', 
        transform=ccrs.PlateCarree(),
        color="blue", 
        markersize=10)

ax.set_extent(extent)
ax.set(title="Location of the Latitude / Longitude Being Used To to Slice Your netcdf Climate Data File")

# Adds continent boundaries to the map
ax.add_feature(cfeature.LAND, edgecolor='black')

ax.gridlines()
plt.show()

# %% Tucson
latitude = grace_dataset['lwe_thickness']['lat'].values[488]
longitude = grace_dataset['lwe_thickness']['lon'].values[276]

print("The current latitude is ", latitude, 'and longitude is -', 180 - longitude)

one_point = lwe2.sel(lat = latitude, lon = longitude)
one_point

f, ax = plt.subplots(figsize=(12, 6))
one_point.plot.line(hue='lat',
                    marker="o",
                    ax=ax,
                    color="grey",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                    )
ax.set(title="Liquid Water Equivalent thickness for Tucson Area (cm)")
plt.xlabel('Days since January 1, 2002')
plt.ylabel('LWE (cm)')
# %% Lower right of Phoenix AMA
# 33.111207, -111.748979
latitude = grace_dataset['lwe_thickness']['lat'].values[492]
longitude = grace_dataset['lwe_thickness']['lon'].values[272]

print("The current latitude is ", latitude, 'and longitude is -', 180 - longitude)

one_point = lwe2.sel(lat = latitude, lon = longitude)
one_point

f, ax = plt.subplots(figsize=(12, 6))
one_point.plot.line(hue='lat',
                    marker="o",
                    ax=ax,
                    color="grey",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                    )
ax.set(title="Liquid Water Equivalent thickness for Lower right of Phoenix AMA (cm)")
#plt.xlabel('Days since January 1, 2002')
plt.ylabel('LWE (cm)')

# %%
# Fort Mojave
# 35.088993, -114.409085
latitude = grace_dataset['lwe_thickness']['lat'].values[500]
longitude = grace_dataset['lwe_thickness']['lon'].values[262]

print("The current latitude is ", latitude, 'and longitude is -', 180 - longitude)

one_point = lwe2.sel(lat = latitude, lon = longitude)
one_point

f, ax = plt.subplots(figsize=(12, 6))
one_point.plot.line(hue='lat',
                    marker="o",
                    ax=ax,
                    color="grey",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                    )
ax.set(title="Liquid Water Equivalent thickness for Fort Mojave Area (cm)")
plt.xlabel('Days since January 1, 2002')
plt.ylabel('LWE (cm)')


#%%
# Yuma
# 32.662849, -114.609830
latitude = grace_dataset['lwe_thickness']['lat'].values[490]
longitude = grace_dataset['lwe_thickness']['lon'].values[261]

one_point = lwe2.sel(lat = latitude, lon = longitude)
one_point

print("The current latitude is ", latitude, 'and longitude is -', 180 - longitude)

f, ax = plt.subplots(figsize=(12, 6))
one_point.plot.line(hue='lat',
                    marker="o",
                    ax=ax,
                    color="grey",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                    )
ax.set(title="Liquid Water Equivalent thickness for Yuma Area (cm)")
plt.xlabel('Days since January 1, 2002')
plt.ylabel('LWE (cm)')

# %% Above Cochise
# 32.854682, -109.677721

latitude = grace_dataset['lwe_thickness']['lat'].values[491]
longitude = grace_dataset['lwe_thickness']['lon'].values[281]

print("The current latitude is ", latitude, 'and longitude is -', 180 - longitude)

one_point = lwe2.sel(lat = latitude, lon = longitude)
one_point

f, ax = plt.subplots(figsize=(12, 6))
one_point.plot.line(hue='lat',
                    marker="o",
                    ax=ax,
                    color="grey",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                    )
ax.set(title="Liquid Water Equivalent thickness for Above Cochise (cm)")
plt.xlabel('Days since January 1, 2002')
plt.ylabel('LWE (cm)')

#%%
# Cochise/Wilcox Area
# 31.663785, -109.698792
latitude = grace_dataset['lwe_thickness']['lat'].values[486]
longitude = grace_dataset['lwe_thickness']['lon'].values[281]

print("The current latitude is ", latitude, 'and longitude is -', 180 - longitude)

one_point = lwe2.sel(lat = latitude, lon = longitude)
one_point

f, ax = plt.subplots(figsize=(12, 6))
one_point.plot.line(hue='lat',
                    marker="o",
                    ax=ax,
                    color="grey",
                    markerfacecolor="blue",
                    markeredgecolor="blue"
                    )
ax.set(title="Liquid Water Equivalent thickness for Cochise/Wilcox Area (cm)")
plt.xlabel('Days since January 1, 2002')
plt.ylabel('LWE (cm)')

# %% Plotting the pixels on the same graphs
# Lower right of Phoenix AMA
# 33.111207, -111.748979
platitude = grace_dataset['lwe_thickness']['lat'].values[492]
plongitude = grace_dataset['lwe_thickness']['lon'].values[272]

phoenix = lwe2.sel(lat = platitude, lon = plongitude)
phoenix

wlatitude = grace_dataset['lwe_thickness']['lat'].values[486]
wlongitude = grace_dataset['lwe_thickness']['lon'].values[281]

wilcox = lwe2.sel(lat = wlatitude, lon = wlongitude)
wilcox

ylatitude = grace_dataset['lwe_thickness']['lat'].values[490]
ylongitude = grace_dataset['lwe_thickness']['lon'].values[261]

yuma = lwe2.sel(lat = ylatitude, lon = ylongitude)
yuma

ulatitude = grace_dataset['lwe_thickness']['lat'].values[500]
ulongitude = grace_dataset['lwe_thickness']['lon'].values[262]

upperco = lwe2.sel(lat = ulatitude, lon = ulongitude)
upperco

f, ax = plt.subplots(figsize=(12, 6))
phoenix.plot.line(hue='lat',
                    # marker="o",
                    ax=ax,
                    color="red",
                    # markerfacecolor="blue",
                    # markeredgecolor="blue"
                    label = 'Phoenix AMA - GW Regulated'
                    )

# wilcox.plot.line(hue='lat',
#                     # marker="o",
#                     ax=ax,
#                     color="blue",
#                     # markerfacecolor="blue",
#                     # markeredgecolor="blue"
#                     label = 'Wilcox - GW Unregulated'
#                     )

# yuma.plot.line(hue='lat',
#                     # marker="o",
#                     ax=ax,
#                     color="green",
#                     # markerfacecolor="blue",
#                     # markeredgecolor="blue"
#                     label = 'Yuma - SW Dominated'
#                     )

# upperco.plot.line(hue='lat',
#                     # marker="o",
#                     ax=ax,
#                     color="teal",
#                     # markerfacecolor="blue",
#                     # markeredgecolor="blue"
#                     label = 'Upper Co (Fort Mojave Area) - Mixed'
#                     )

ax.set(title="Individual GRACE Pixels from the Baseline")
ax.legend()
ax.grid(zorder = 0)
plt.xlabel('Year')
plt.ylabel('LWE (cm)')

# plt.savefig('Individual_Pixels_Phoenix_Wilcox')

# %%
platitude = grace_dataset['lwe_thickness']['lat'].values[492]
plongitude = grace_dataset['lwe_thickness']['lon'].values[272]

print("The current latitude is ", platitude, 'and longitude is -', 180 - plongitude)

phoenix = lwe2.sel(lat = platitude, lon = plongitude)
phoenix

phoenix.plot()

phoenix_df = pd.DataFrame(phoenix)
phoenix_df

phoenix_df = phoenix_df.reset_index()
phoenix_df

phoenix_df['index'] = datetimeindex
phoenix_df

phoenix_df.set_index('index', inplace=True)
phoenix_df

# Extract the year from the date column and create a new column year
phoenix_df['year'] = pd.DatetimeIndex(phoenix_df.index).year
phoenix_df.head()

phoenix_df_year = pd.pivot_table(phoenix_df, index=["year"], values=[0], dropna=False, aggfunc=np.mean)
phoenix_df_year

# %%
ylatitude = grace_dataset['lwe_thickness']['lat'].values[490]
ylongitude = grace_dataset['lwe_thickness']['lon'].values[261]

print("The current latitude is ", platitude, 'and longitude is -', 180 - plongitude)

yuma = lwe2.sel(lat = ylatitude, lon = ylongitude)
yuma

yuma.plot()

yuma_df = pd.DataFrame(yuma)
yuma_df

yuma_df = yuma_df.reset_index()
yuma_df

yuma_df['index'] = datetimeindex
yuma_df

yuma_df.set_index('index', inplace=True)
yuma_df

# Extract the year from the date column and create a new column year
yuma_df['year'] = pd.DatetimeIndex(yuma_df.index).year
yuma_df.head()

yuma_df_year = pd.pivot_table(yuma_df, index=["year"], values=[0], dropna=False, aggfunc=np.mean)
yuma_df_year

# %%
wlatitude = grace_dataset['lwe_thickness']['lat'].values[486]
wlongitude = grace_dataset['lwe_thickness']['lon'].values[281]

wilcox = lwe2.sel(lat = wlatitude, lon = wlongitude)
wilcox

print("The current latitude is ", wlatitude, 'and longitude is -', 180 - wlongitude)

wilcox.plot()

wilcox_df = pd.DataFrame(wilcox)
wilcox_df

wilcox_df = wilcox_df.reset_index()
wilcox_df

wilcox_df['index'] = datetimeindex
wilcox_df

wilcox_df.set_index('index', inplace=True)
wilcox_df

# Extract the year from the date column and create a new column year
wilcox_df['year'] = pd.DatetimeIndex(wilcox_df.index).year
wilcox_df.head()

wilcox_df_year = pd.pivot_table(wilcox_df, index=["year"], values=[0], dropna=False, aggfunc=np.mean)
wilcox_df_year

#%%
ulatitude = grace_dataset['lwe_thickness']['lat'].values[500]
ulongitude = grace_dataset['lwe_thickness']['lon'].values[262]

upperco = lwe2.sel(lat = ulatitude, lon = ulongitude)
upperco

print("The current latitude is ", wlatitude, 'and longitude is -', 180 - wlongitude)

wilcox.plot()

wilcox_df = pd.DataFrame(wilcox)
wilcox_df

wilcox_df = wilcox_df.reset_index()
wilcox_df

wilcox_df['index'] = datetimeindex
wilcox_df

wilcox_df.set_index('index', inplace=True)
wilcox_df

# Extract the year from the date column and create a new column year
wilcox_df['year'] = pd.DatetimeIndex(wilcox_df.index).year
wilcox_df.head()

wilcox_df_year = pd.pivot_table(wilcox_df, index=["year"], values=[0], dropna=False, aggfunc=np.mean)
wilcox_df_year

# %%
# ---- Plotting Averages Based off Shape File Mask ----
# Check the cooridnate systems
mask = counties
print("mask crs:", counties.crs)
print("data crs:", lwe2.rio.crs)

# %% Clipping based off the mask (not weighted)
clipped = lwe2.rio.clip(mask.geometry, mask.crs)
clipped.plot()
# %% Write the clipped file into .tif
#lwe2.rio.to_raster(r"testGRACE_time.tif")

#clipped.rio.to_raster(r"clipped_GRACE_counties.tif")
#print(".tif written")
# %%
fig, ax = plt.subplots(figsize=(6, 6))
#lwe2[0,:,:].plot()
clipped[0,:,:].plot()
mask.boundary.plot(ax=ax)
#ax.set_ylim(31,37)
#ax.set_xlim([-115, -109])
ax.set_title("Shapefile Crop Extent",
             fontsize=16)
plt.show()

# %%
clipped['time'] = datetimeindex

# %%
clipped_mean = clipped.mean(("lon","lat"))
clipped_mean

# %%
global_mean = lwe2.mean(("lon","lat"))
global_mean

# %%
cm_df = pd.DataFrame(clipped_mean)
cm_df.info()

# %%
cm_df = cm_df.reset_index()
cm_df
# %%
cm_df['index'] = datetimeindex
cm_df

# %%
cm_df.set_index('index', inplace=True)
cm_df

# %%
# Extract the year from the date column and create a new column year
cm_df['year'] = pd.DatetimeIndex(cm_df.index).year
cm_df.head()

# %%
cm_df_year = pd.pivot_table(cm_df, index=["year"], values=[0], dropna=False, aggfunc=np.mean)
cm_df_year

# %%
clipped_mean.plot()
# %%
cm_df_year.plot(label="AZ mean (not weighted)")
#global_mean.plot(label="global Mean")
plt.legend()

# %% Plotting single points with the state average
f, ax = plt.subplots(figsize=(12, 6))
ax.plot(cm_df_year, color='#2F2F2F', label='Arizona Average')
ax.plot(phoenix_df_year, color='red', label = ' Pixel in Phoenix AMA - GW Regulated')
ax.plot(yuma_df_year, color='green', label = ' Pixel in Yuma Area - GW Unregulated')


ax.set(title="Individual GRACE Pixels Change from the 2004-2009 Baseline")
ax.legend()
ax.grid(zorder = 0)
plt.xlabel('Year')
plt.ylabel('Change in LWE (cm)')

# %%
f, ax = plt.subplots(figsize=(12,6))
ax.plot(phoenix_df_year, color='red')
ax.set(title="Change in Liquid Water Equivalent from the 2004-2009 Baseline in Phoenix AMA")
# ax.legend()
ax.grid(zorder = 0)
plt.xlabel('Year')
plt.ylabel('LWE Change (cm)')
ax.set_xlim(2002,2020)


# %%
# Plot all of the georegions and state average
ds = grace_yearlyavg
name = "Annual Change from the 2004-2009 Baseline"
ylabel = "Liquid Water Equivalent (cm)"
minyear=2002
maxyear=2020
min_y = -15
max_y = 7
fsize = 14

# For the actual figure
fig, ax = plt.subplots(2,2,figsize=(24,12))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.09)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0,0].plot(ds['Regulated with CAP'], label='Regulated with CAP', color=c_2) 
ax[0,0].plot(ds['Regulated without CAP'], label='Regulated without CAP', color=c_3) 
ax[1,1].plot(ds['Lower Colorado River - SW Dominated'], color=c_4, label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds['Upper Colorado River - Mixed'], color=c_5, label='Upper Colorado River - Mixed')
ax[0,1].plot(ds['Norh - Mixed'], color=c_10, label='North - Mixed')
ax[0,1].plot(ds['Central - Mixed'], color=c_11, label='Central - Mixed')
ax[1,0].plot(ds['Northwest - GW Dominated'], color=c_7, label='Northwest - GW Dominated')
ax[1,0].plot(ds['Northeast - GW Dominated'], color=c_9, label='Northeast - GW Dominated')
ax[1,0].plot(ds['South central - GW Dominated'], color=c_8, label='South central - GW Dominated')
ax[1,0].plot(ds['Southeast - GW Dominated'], color=c_6, label='Southeast - GW Dominated')
#Plotting Arizona Average
ax[0,0].plot(cm_df_year, color='#2F2F2F', label='Arizona Average')
ax[0,1].plot(cm_df_year, color='#2F2F2F', label='Arizona Average')
ax[1,0].plot(cm_df_year, color='#2F2F2F', label='Arizona Average')
ax[1,1].plot(cm_df_year, color='#2F2F2F', label='Arizona Average')

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

# Drought Year Shading
a = 2011
b = 2015.999
c = 2018.001
d = 2018.999
e = 2006
f = 2007.999
drought_color = '#ffa6b8'
wet_color = '#b8d3f2'

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

ax[0,0].legend(loc = [0.05, 0.15], fontsize = fsize)
ax[0,1].legend(loc = [0.05, 0.05], fontsize = fsize)
ax[1,0].legend(loc = [0.05, 0.04], fontsize = fsize)
ax[1,1].legend(loc = [0.05, 0.18], fontsize = fsize)

# plt.savefig(outputpath+name+'_AZavg_drought')

# %%
# Plot ag verus non-ag with the state
ds = grace_yearlyavg
name = "Annual Change from the 2004-2009 Baseline - Agriculture Regions"
ylabel = "Liquid Water Equivalent (cm)"
minyear=2002
maxyear=2020
min_y = -13
max_y = 5
fsize = 14

# For the actual figure
fig, ax = plt.subplots(figsize=(12,9))
#fig.tight_layout()
fig.suptitle(name, fontsize=20, y=0.91)
fig.supylabel(ylabel, fontsize = 14, x=0.07)
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
# ax.plot(ds['Ag'], label='Agriculture', color='green') 
ax.plot(ds['Non_Ag'], label='Non-Agriculture', color='#cb9859',lw=2) 

#Plotting Arizona Average
ax.plot(cm_df_year, '-.',color='#2F2F2F', label='Arizona Average', zorder=0)

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
ax.grid(True)
#ax[0,0].set(title=name, xlabel='Year', ylabel='Change from Baseline (cm)')
#ax[0,0].set_title(name, loc='right')
#ax[1,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='top', fontsize = fsize)

# Drought Year Shading
a = 2010.5
b = 2015.5
c = 2017.5
d = 2018.5
e = 2005.5
f = 2008.5
p = 2002.5
q = 2003.5
drought_color = '#ffa6b8'
wet_color = '#b8d3f2'

ax.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
ax.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
ax.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
ax.axvspan(p, q, color=drought_color, alpha=0.5, lw=0)



# Wet years (2005 and 2010)
g = 2004.5
h = 2009.5
ax.axvspan(g, e, color=wet_color, alpha=0.5, lw=0, label="Wet Years")
ax.axvspan(h, a, color=wet_color, alpha=0.5, lw=0)

ax.legend(loc = [0.05, 0.15], fontsize = fsize)

plt.savefig(outputpath+name+'_AZavg_drought_noag')
# %% Now let's run some statistics

