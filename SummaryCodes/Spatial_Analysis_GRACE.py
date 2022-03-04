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


# %% ---- Remapping using EASYMORE Package ----
# Prepping the data
# reprojecting coordinate system
#reproject = counties.to_crs(epsg=4326)
reproject = georeg.to_crs(epsg=4326)
reproject.crs
# %%
reproject.to_file("georeg_reproject.shp")
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
esmr.case_name                = 'easymore_GRACE_georeg'              
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
esmr.target_shp = 'georeg_reproject_fixed.shp'
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
filename = 'easymore_GRACE_georeg_remapped_lwe_thickness__2002-04-18-00-00-00.csv'
filepath = os.path.join(outputpath, filename)
grace_remapped = pd.read_csv(filepath)
grace_remapped.head()
# %% Gotta fix those headers
ID_key = shp_target[['GEO_Region', 'ID_t']]
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
georeg_list = ID_key['GEO_Region'].values.tolist()
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
#grace_yearlyavg['year'] = pd.to_numeric(grace_yearlyavg['year'])
#grace_yearlyavg['year'] = grace_yearlyavg['year'].astype(int)
grace_yearlyavg = grace_yearlyavg.set_index('year', inplace=True)
grace_yearlyavg

# %%
grace_yearlyavg.plot()

#%%
ds = grace_yearlyavg
name = "Average Depth to Water"
minyear=2002
maxyear=2020
min_y = -15
max_y = 7

# Plot all of them
fig, ax = plt.subplots(2,2,figsize=(32,18))
#fig.tight_layout()
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0,0].plot(ds['Regulated with CAP'], label='Regulated with CAP', color="#d7191c") 
ax[0,0].plot(ds['Regulated without CAP'], label='Regulated without CAP', color='#e77a47') 
ax[1,1].plot(ds['Lower Colorado River - SW Dominated'], color='#2cbe21', label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds['Upper Colorado River - Mixed'], color='#2f8c73', label='Upper Colorado River - Mixed')
ax[0,1].plot(ds['Norh - Mixed'], color='#41bf9e', label='North - Mixed')
ax[0,1].plot(ds['Central - Mixed'], color='#7adec4', label='Central - Mixed')
ax[1,0].plot(ds['Northwest - GW Dominated'], color='#165782', label='Northwest - GW Dominated')
ax[1,0].plot(ds['Northeast - GW Dominated'], color='#1f78b4', label='Northeast - GW Dominated')
ax[1,0].plot(ds['South central - GW Dominated'], color='#229ce8', label='South central - GW Dominated')
ax[1,0].plot(ds['Southeast - GW Dominated'], color='#6db7e8', label='Southeast - GW Dominated')
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
ax[0,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='bottom')
ax[0,0].legend(loc = [0.1, 0.20])
ax[0,1].legend(loc = [0.1, 0.05])
ax[1,0].legend(loc = [0.1, 0.05])
ax[1,1].legend(loc = [0.1, 0.20])

# %% Write a .csv for now for graphing later
grace_remapped.to_csv('../MergedData/Output_files/grace_remapped.csv')
grace_yearlyavg.to_csv('../MergedData/Output_files/grace_remapped_yearly.csv')

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
clipped[2,:,:].plot(cmap='viridis')
mask.boundary.plot()
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

# %%
# Plot all of them
fig, ax = plt.subplots(2,2,figsize=(29,18))
#fig.tight_layout()
#ax[1,1].plot(ds['Reservation'], label='Reservation', color='#8d5a99')
ax[0,0].plot(ds['Regulated with CAP'], label='Regulated with CAP', color="#d7191c") 
ax[0,0].plot(ds['Regulated without CAP'], label='Regulated without CAP', color='#e77a47') 
ax[1,1].plot(ds['Lower Colorado River - SW Dominated'], color='#2cbe21', label='Lower Colorado River - SW Dominated')
ax[0,1].plot(ds['Upper Colorado River - Mixed'], color='#2f8c73', label='Upper Colorado River - Mixed')
ax[0,1].plot(ds['Norh - Mixed'], color='#41bf9e', label='North - Mixed')
ax[0,1].plot(ds['Central - Mixed'], color='#7adec4', label='Central - Mixed')
ax[1,0].plot(ds['Northwest - GW Dominated'], color='#165782', label='Northwest - GW Dominated')
ax[1,0].plot(ds['Northeast - GW Dominated'], color='#1f78b4', label='Northeast - GW Dominated')
ax[1,0].plot(ds['South central - GW Dominated'], color='#229ce8', label='South central - GW Dominated')
ax[1,0].plot(ds['Southeast - GW Dominated'], color='#6db7e8', label='Southeast - GW Dominated')

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
ax[0,0].set_ylabel("Change from 2004-2009 Baseline (cm)", loc='bottom')
ax[0,0].legend(loc = [0.1, 0.20])
ax[0,1].legend(loc = [0.1, 0.05])
ax[1,0].legend(loc = [0.1, 0.05])
ax[1,1].legend(loc = [0.1, 0.20])
# %%
