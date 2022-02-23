# === GRACE Spatial Analysis Script ===
# written by Danielle Tadych
# The purpose of this script is to analyze GRACE Data for Arizona by points and shapes
#  - Importing packages: Line 6
#  - Reading in files: Line 37
#  - EASYMORE Remapping using a shapefile: Line 56
#     *Note: in order for this package to work
#               > the projections need to be in espg4326, exported, re-read in
#               > the time variable for nc needs to be datetime
#               > Value error -> run the "fix geometries" tool in qgis
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
# %%
grace_yearlyavg = grace_yearlyavg.set_index("year")
grace_yearlyavg
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
mask.plot(ax=ax)
clipped[0,:,:].plot()
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
# --- Computing zonal stats (weighed average) ---

# This is a tutorial that runs but the means are none
# https://gis.stackexchange.com/questions/363120/computing-annual-spatial-zonal-statistics-of-a-netcdf-file-for-polygons-in-sha


# load and read shp-file with geopandas
#shp_fo = r'../path/to/shp_file.shp'
#shp_df = gpd.read_file(shp_fo)
shp_df = counties

# load and read netCDF-file to dataset and get datarray for variable
#nc_fo = r'../path/to/netCDF_file.nc'
#nc_ds = xr.open_dataset(nc_fo)
#nc_var = nc_ds['var_name']
nc_ds = grace_dataset
nc_var = grace_dataset['lwe_thickness']

# get all years for which we have data in nc-file
years = nc_ds['time'].values
# %%
# get affine of nc-file with rasterio
af = rasterio.open(r'../../GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc').transform
print(af)
#%%
# go through all years
for year in years:
    # get values of variable pear year
    nc_arr = nc_var.sel(time=year)
    nc_arr_vals = nc_arr.values
    # go through all geometries and compute zonal statistics
    for i in range(len(shp_df)):
        print(zonal_stats(shp_df.geometry, nc_arr_vals, affine=af, stats="mean"))

# %%
# Following this tutorial - didn't work though
# https://automating-gis-processes.github.io/CSC/notebooks/L5/zonal-statistics.html
#dem=rasterio.open("clipped_GRACE_counties.tif")
#dem
# %%
ax = counties.plot(facecolor='None', edgecolor='red', linewidth=2)
show((dem, 1), ax=ax)
# %%
dem.info()
# %%
counties = counties.to_crs(crs=dem.crs)
type(counties)
# %%
array = dem.read(1)

af = dem.transform
# %%
zs_counties = zonal_stats(counties, lwe2)
# %%
# Trying from xarray
weights = np.cos(np.deg2rad(lwe2.lat))
weights.name = "weights"
weights
# %%
global_weighted = lwe2.weighted(weights)
global_weighted
# %%
global_weighted_mean = global_weighted.mean(("lon","lat"))
global_weighted_mean

# %%
global_weighted_mean.plot(label="global")
#clipped.mean(("lon","lat")).plot(label="Arizona")
plt.legend()
# %% OKAY, Going to try from scratch
# Polygonize raster - did this in qgis by polygonizing Band 1, 2, and 202, clipping AZ shape, then union of all 3
filename = "Grace_PixelShapes.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles/GRACE_Scratchfiles', filename)
grace_shape = gp.read_file(filepath)
grace_shape.plot()
# %% Load in new mask
filename = "pimacounty.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles', filename)
mask = gp.read_file(filepath)
mask.plot()
# %%
mask = mask[['NAME', 'geometry']]
mask
# %% 
# Calculate area of each GRACE pixel
grace_shape['pixel_area'] = grace_shape.geometry.area #/10000 for hectares
print(grace_shape.head())
# %%
grace_shape = grace_shape[['geometry','pixel_area']]
grace_shape
# %% Clip based off mask
grace_clip = gp.clip(grace_shape, mask)
# %%
# Overlay shapefile of mask
overlay = gp.overlay(grace_clip, mask, how='union')
overlay
# %% Calculate new area of the pixels
overlay['overlay_area'] = overlay.geometry.area
overlay

# %%
# dataframe of weights = new shape area/ Total pixel area
overlay['weights'] = overlay['overlay_area']/overlay['pixel_area']
overlay
# %%
overlay.plot()
#%%
overlay.to_file("pima_GRACE_Overlay.shp")

# %% Going to Rasterize the shapefile in qgis then read it back in
overlay_raster=rasterio.open("clipped_GRACE_counties.tif")
#%%
# lwe = weights x original grace value
grace_mask = lwe2.rio.clip(mask.geometry, mask.crs)
grace_mask.plot()
# %%
Grace_weighted = grace_mask['lwe_thickness'] * overlay['weights']
Grace_weighted
# %% Way to use rasterstats with .nc data
#https://gis.stackexchange.com/questions/363120/computing-annual-spatial-zonal-statistics-of-a-netcdf-file-for-polygons-in-sha

# get all years for which we have data in nc-file
years = lwe2.resample('Y').sum()

# %%
# get affine of nc-file with rasterio
affine = rasterio.open(r'../../GRACE/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc').transform
# %%
# go through all years
for year in years:
    # get values of variable pear year
    nc_arr = lwe2.sel(time=year)
    nc_arr_vals = nc_arr.values
    # go through all geometries and compute zonal statistics
    for i in range(len(mask)):
        print(rstats.zonal_stats(mask.geometry, nc_arr_vals, affine=affine, stats="mean min max"))
        print('')
# %%
print(lwe2.time)
# %%
nc_arr = lwe2.sel(time='2002-04-18T00:00:00.000000000')
nc_arr_vals = nc_arr.values
nc_arr_vals
# %%
test_mean = rstats.zonal_stats(mask.geometry, nc_arr_vals, affine=affine, stats='mean')
print(test_mean)
# %%
