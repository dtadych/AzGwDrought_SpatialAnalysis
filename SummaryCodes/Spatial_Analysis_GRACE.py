# === GRACE Spatial Analysis Script ===
# written by Danielle Tadych
# The purpose of this script is to analyze GRACE Data for Arizona by points and shapes
# Lines 6->
# %%
from calendar import calendar
from itertools import count
import os
from typing import Mapping
#import affine
from geopandas.tools.sjoin import sjoin
import matplotlib
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
import rasterstats as rstats
#from xrspatial import zonal_stats

# %% Read in the file
filename = 'CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc'
datapath = '../../GRACE'
print(datapath)
outputpath = '../MergedData/Output_files/'

grace_dataset = xr.open_dataset(datapath+'/'+filename)

grace_dataset

# %% Read in the mask shapefile
filename = "Georegions_AGU.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Output_files', filename)
georeg = gp.read_file(filepath)

# %%
filename = "AZ_counties.shp"
filepath = os.path.join('/Users/danielletadych/Documents/PhD_Materials/github_repos/AzGwDrought_SpatialAnalysis/MergedData/Shapefiles', filename)
counties = gp.read_file(filepath)

# %%
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
# This wrote a tif!!!

# %% Plot the new geotif
da = xr.open_rasterio("testGRACE_time.tif")
#transform = cartopy.Affine.from_gdal(*da.attrs["transform"]) # this is important to retain the geographic attributes from the file
da = 
# %%
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.imshow(da.variable.data[0])
plt.show()

# %%
# Plot!
crs=ccrs.PlateCarree()
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection=crs)
ax.coastlines(resolution='10m', alpha=0.1)
ax.contourf(x, y, da.variable.data[0], cmap='Greys')
ax.set_extent([lon_min, lon_max, lat_min, lat_max])
# Grid and Labels
gl = ax.gridlines(crs=crs, draw_labels=True, alpha=0.5)
gl.xlabels_top = None
gl.ylabels_right = None
xgrid = np.arange(lon_min-0.5, lon_max+0.5, 1.)
ygrid = np.arange(lat_min, lat_max+1, 1.)
gl.xlocator = mticker.FixedLocator(xgrid.tolist())
gl.ylocator = mticker.FixedLocator(ygrid.tolist())
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 14, 'color': 'black'}
gl.ylabel_style = {'size': 14, 'color': 'black'}
plt.show()

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
plt.xlabel('Days since January 1, 2002')
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
# ---- Plotting weighted Averages Based off Shape File Mask ----
# Check the cooridnate systems
mask = counties
print("mask crs:", counties.crs)
print("data crs:", lwe2.rio.crs)

# %% Clipping based off the mask
clipped = lwe2.rio.clip(mask.geometry, mask.crs)
clipped.plot()
# %% Write the clipped file into .tif
#lwe2.rio.to_raster(r"testGRACE_time.tif")

#clipped.rio.to_raster(r"clipped_GRACE_counties.tif")
#print(".tif written")

# %%
fig, ax = plt.subplots(figsize=(6, 6))

mask.plot(ax=ax)

ax.set_title("Shapefile Crop Extent",
             fontsize=16)
plt.show()

# %%
clipped['time'] = datetimeindex
# %%
clipped.rio.crs
# %%
# Following this tutorial - didn't work though
# https://automating-gis-processes.github.io/CSC/notebooks/L5/zonal-statistics.html
dem=rasterio.open("clipped_GRACE_counties.tif")
dem
# %%
ax = counties.plot(facecolor='None', edgecolor='red', linewidth=2)
show((dem, 1), ax=ax)
# %%
dem.info()
# %%
counties = counties.to_crs(crs=dem.crs)
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
