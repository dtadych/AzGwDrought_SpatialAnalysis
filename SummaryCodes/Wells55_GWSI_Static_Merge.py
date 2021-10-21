# The purpose of this script is combine the GWSI and Wells55 databases into one
# Written by Danielle Tadych
# Goals:
# - Create columns in each respective database specifying its origin
# - Find column they have in common
# - Merge based on that column
# - Make GWSI wells the overriding database and fill in the gaps with Wells55
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
shapedir = '../MergedData/Shapefiles'
#wellfilename = "Well_Registry__Wells55_.shp"
wellfilename = "Well_Registry__Wells55_.shp"
Wellfp = os.path.join(shapedir, wellfilename)
wells55shape = gp.read_file(Wellfp)

GWSI_fn = "GWSI_SITES.shp"
Wellfp = os.path.join(shapedir, GWSI_fn)
GWSIshape = gp.read_file(Wellfp)
# %%
GWSIshape.info()
# %%
wells55shape.info()
# %% Check that they're in the same coordinate system
print(GWSIshape.crs, wells55shape.crs)

# %%
# Preliminary Plot to check shapefiles
fig, ax = plt.subplots()
GWSIshape.plot(ax = ax, label="GWSI")
wells55shape.plot(ax = ax, label="Wells55")
ax.set_title("GWSI and Wells55 Preliminary Plot")
plt.legend()

# %% Making copies of the databases so I don't overright the originals
gwsi_gdf = GWSIshape
wells55_gdf = wells55shape

# This was to filter by driller license number but that's dumb
#wells55_gdf = wells55_gdf[wells55_gdf['DLIC_NUM'].notna()]
#wells55_gdf.info()

# ---- Adding Database Source Columns to both ----
wells55_gdf["Original_DB"] = 'Wells55'
gwsi_gdf["Original_DB"] = 'GWSI'
wells55_gdf.head()
# %%
gwsi_gdf.head()

# %% Fixing the date so that 1/1/70 in Wells55 is replaced with NAN
# https://stackoverflow.com/questions/29247712/how-to-replace-a-value-in-pandas-with-nan
wells55_gdf['INSTALLED'] = wells55_gdf['INSTALLED'].replace(['1970-01-01'], np.NaN)
wells55_gdf['INSTALLED'].unique()


# %% ---- Merging Both databases ----

# Merge wells55 'REGISTRY_I' with GWSI 'REG_ID'
# need to use how = left
#  - more info here https://www.datasciencemadesimple.com/join-merge-data-frames-pandas-python/
#  - and here regarding default options for merge
#    https://stackabuse.com/how-to-merge-dataframes-in-pandas/#mergedataframesusingmerge

# %% Changing REG_ID in GWSI to REGISTRY_I
gwsi_gdf.rename(columns={'REG_ID':'REGISTRY_I'}, inplace=True)

# -- Stop here if you want to filter the wells55 database for specific uses
# -- skip to line 98
# %%
#Wells55_GWSI_MasterDB = wells55_gdf.merge(gwsi_gdf, suffixes=['_wells55','_gwsi'], how="outer", left_on="REGISTRY_I", right_on="REG_ID")
Wells55_GWSI_MasterDB = wells55_gdf.merge(gwsi_gdf, suffixes=['_wells55','_gwsi'], how="outer", 
                                          left_on=["REGISTRY_I", 'WELLTYPE', 'WELL_DEPTH', 'geometry', 'Original_DB'],
                                          right_on=["REGISTRY_I", 'WELL_TYPE', 'WELL_DEPTH', 'geometry', 'Original_DB'])
print(Wells55_GWSI_MasterDB.info())

# %% combine registry ID and Site ID so in timeseries graphs every well has an ID
Wells55_GWSI_MasterDB['Combo_ID'] = Wells55_GWSI_MasterDB.REGISTRY_I.combine_first(Wells55_GWSI_MasterDB.SITE_ID)
Wells55_GWSI_MasterDB.info()
# %%
# Now plot the new master db
fig, ax = plt.subplots()
#gwsi_gdf.plot(ax = ax, label="GWSI")
#wells55_gdf.plot(ax = ax, label="Wells55")
Wells55_GWSI_MasterDB.plot(ax=ax, label="Master Database")
ax.set_title("Check the merged database")
plt.legend()
plt.savefig('../MergedData/Output_files/{0}.png'.format(type), bbox_inches='tight')

# %%
# Export all the ish
#Wells55_GWSI_MasterDB.to_file("Master_ADWR_Database_v3.shp")
Wells55_GWSI_MasterDB.to_csv('../MergedData/Output_files/Master_ADWR_database_datesfixed.csv')
Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database_datesfixed.shp')
# %%
# --- Filter wells55 and make new MasterDB
# First, deleting the cancelled wells
wells55_nocanc = wells55_gdf[wells55_gdf.WELL_CANCE != 'Y']
wells55_nocanc.info()
