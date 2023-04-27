# ---- Processing NHD Stream Shapefile ----
# written by Danielle Tadych
# 4/26/23

# The purpose of this code is to make the NHD stream shapefile less of a monster for graphs

#%%


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
from scipy.stats import kendalltau, pearsonr, spearmanr
#import rasterstats as rstats
#from xrspatial import zonal_stats
import easymore
import glob
import scipy.stats as sp
import pymannkendall as mk

# %% Data paths
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/'

filename_georeg = 'NHD_Important_Rivers.shp'
filepath = os.path.join(shapepath, filename_georeg)
nhd_rivers = gp.read_file(filepath)
# %%
nhd_rivers

# %%
to_dissolve = nhd_rivers[['GNIS_Name','geometry']]
dissolved = to_dissolve.dissolve(by='GNIS_Name')
dissolved

#%% Check to see it worked
# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))
dissolved.reset_index().plot(column='GNIS_Name',
                            ax=ax)
ax.set_axis_off()
plt.axis('equal')
plt.show() 
# %% Calculate stream lengths

dissolved['length'] = dissolved['geometry'].length
dissolved.head()
# %%
dissolved.describe()
# %%
narrowed = dissolved[(dissolved.length > 0.23)]
narrowed
# %%
narrowed.plot()
# %%
narrowed.describe()
# %%
more_narrowing = narrowed[(narrowed.length > 1.9)]
more_narrowing
# %%
more_narrowing.plot()
# %%
more_narrowing = more_narrowing.reset_index()
more_narrowing
#%%
more_narrowing.loc[[8],'geometry'].plot()
# %%
ImportantStreams = more_narrowing
ImportantStreams = ImportantStreams.drop([0,3,8,9,11])
ImportantStreams
# %%
ImportantStreams.plot()
# %%
more_narrowing.to_file('../MergedData/Output_files/Narrowed_Important_SWFlowlines.shp')
# %%
