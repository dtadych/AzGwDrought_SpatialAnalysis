# === GRACE Spatial Analysis Script ===
# written by Danielle Tadych
# The purpose of this script is to analyze GRACE Data for Arizona by points and shapes
# Lines 6->
# %%
from calendar import calendar
from itertools import count
import os
from typing import Mapping
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
import rasterstats

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
clipped = lwe2.rio.clip(mask.geometry, counties.crs)
clipped.plot()
# %% Write the clipped file into .tif
#lwe2.rio.to_raster(r"testGRACE_time.tif")

#clipped.rio.to_raster(r"clipped_GRACE_counties.tif")
#print(".tif written")

# %%


# %%
fig, ax = plt.subplots(figsize=(6, 6))

mask.plot(ax=ax)

ax.set_title("Shapefile Crop Extent",
             fontsize=16)
plt.show()
