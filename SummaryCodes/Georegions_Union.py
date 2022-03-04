# The only reason this code exists is because the last shape was having serious issues union-ing in QGIS
#Author: Danielle Tadych, 2/17/2022

# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import geopandas as gp

# %% 
# ----- Import the Data and Shapefiles with Geometries -----
union_dir = '../MergedData/Shapefiles/Final_Georegions/Union'
#wellfilename = "Well_Registry__Wells55_.shp"
Union_fn = "Ten_123456789.shp"
Union_fp = os.path.join(union_dir, Union_fn)
Union_Ten = gp.read_file(Union_fp)
Union_Ten
Union_Ten.plot()
# %%
diss_dir = '../MergedData/Shapefiles/Final_Georegions/Dissolved'
Central_fn = "S_Mixed_Dissolved.shp" #Note: used to be called South
Central_fp = os.path.join(diss_dir, Central_fn)
Central = gp.read_file(Central_fp)
Central
# %% Final Union
print(Union_Ten.crs,Central.crs) # Check the cooridnate systems
# %%
Central = Central.to_crs(epsg=4269)
print(Central.crs)
Central.plot()

# %%
# gwsi_gdf.rename(columns={'REG_ID':'REGISTRY_I'}, inplace=True)

Central.rename(columns={'OBJECTID':'_OBJECTID','BASIN_NAME':'_BASIN_NAME','NAME_ABBR':'_NAME_ABBR','GEO_Region':'_GEO_Region'}, inplace=True)
Central.info()

# %%
Final_Union = gp.overlay(Union_Ten, Central, how='union')
Final_Union.info()
Final_Union.plot() # Check that it actually worked
# %%
Final_Union.to_file('../MergedData/Output_files/Final_Union.shp')

# %%
