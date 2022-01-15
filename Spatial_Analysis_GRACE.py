# === GRACE Spatial Analysis Script ===
# written by Danielle Tadych
# The purpose of this script is to analyze GRACE Data
# %%
import os
from geopandas.tools.sjoin import sjoin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime as dt
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
import xarray as xr
import rioxarray as rxr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# %% Read in the file
filename = 'CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc'
datapath = '../GRACE'
print(datapath)
outputpath = '../MergedData/Output_files/'

grace_dataset = xr.open_dataset(datapath+'/'+filename)

#%%
grace_dataset

# %%
grace_crs = grace_dataset.rio.crs
grace_crs

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

# %%
key = 400
longitude = -110.911789
latitude = 32.25346
print("Long, Lat values:", longitude, latitude)

# %% Project the grace data
grace_projected = grace_dataset.to_crs('+proj=robin')

# %% Lat an longitude of Tucson
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

# %%
# Slice the data spatially using a single lat/lon point
one_point = grace_dataset["lwe_thickness"].sel(lat=latitude,
                                               lon=longitude)
one_point

#%% From online
# The (online) url for a MACAv2 dataset for max monthly temperature
data_path = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_macav2metdata_tasmax_BNU-ESM_r1i1p1_historical_1950_2005_CONUS_monthly.nc"

max_temp_xr  = xr.open_dataset(data_path)  
# View xarray object
max_temp_xr