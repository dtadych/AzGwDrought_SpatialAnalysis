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
outputpath = '../MergedData/Output_files/'

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

# %% combine registry ID and Site ID so in timeseries graphs every well has an ID
gwsi_gdf['Combo_ID'] = gwsi_gdf.REGISTRY_I.combine_first(gwsi_gdf.SITE_ID)
gwsi_gdf.info()

# %%
wells55_gdf['Combo_ID'] = wells55_gdf['REGISTRY_I']
wells55_gdf.info()

# -- Stop here if you want to filter the wells55 database for specific uses
# -- skip to line 98
# %%
#Wells55_GWSI_MasterDB = wells55_gdf.merge(gwsi_gdf, suffixes=['_wells55','_gwsi'], how="outer", left_on="REGISTRY_I", right_on="REG_ID")
Wells55_GWSI_MasterDB = pd.merge(gwsi_gdf, wells55_gdf, suffixes=['_gwsi','_wells55'], how="outer", 
#                                          left_on=["REGISTRY_I", 'WELL_DEPTH', 'geometry', 'Original_DB'],
#                                          right_on=["REGISTRY_I", 'WELL_DEPTH', 'geometry', 'Original_DB'])
                                          on=['OBJECTID','Combo_ID',"REGISTRY_I", 'WELL_DEPTH', 'geometry', 'Original_DB'])
print(Wells55_GWSI_MasterDB.info())

# %% Looking for flaming duplicates
print(Wells55_GWSI_MasterDB[Wells55_GWSI_MasterDB['Combo_ID'].duplicated()])

# %%
test = Wells55_GWSI_MasterDB.groupby(['Combo_ID']).first()
test.info()

# %%
test = test.reset_index()
print(test[test['Combo_ID'] == '921247'].loc[:,['geometry','Original_DB']])

# %% Check for duplicates again
print(test[test['Combo_ID'].duplicated()])

# %% hooray! No duplicates, now to re-create all those damn files
Wells55_GWSI_MasterDB = test

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
Wells55_GWSI_MasterDB.to_csv('../MergedData/Output_files/Master_ADWR_database_noduplicates.csv')
Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database_noduplicates.shp')
# %%
# --- Filter wells55 and make new MasterDB
# First, deleting the cancelled wells
wells55_nocanc = wells55_gdf[wells55_gdf.WELL_CANCE != 'Y']
wells55_nocanc.info()

#%%
# No classified mineral or exploratory wells
wells55_nomin = wells55_nocanc[wells55_nocanc.WELL_TYPE_ != 'OTHER']
wells55_nomin.info()

# %% Now filter by drill log exisence (not just filed) so we can see which water wells are actually a thing
#wells55_water = wells55_nomin
#wells55_water['DRILL_LOG'] = wells55_water['DRILL_LOG'].replace(None, np.NaN)
wells55_water = wells55_nomin[wells55_nomin['DRILL_LOG'].notna()]
wells55_water.info()

#df = df[df['EPS'].notna()]

# %% Export both the non-cancelled wells and the water wells
#  First the non-cancelled wells
Wells55_GWSI_MasterDB = pd.merge(gwsi_gdf, wells55_nocanc, suffixes=['_gwsi','_wells55'], how="outer", 
                                          on=['OBJECTID','Combo_ID',"REGISTRY_I", 'WELL_DEPTH', 'geometry', 'Original_DB'])

print(Wells55_GWSI_MasterDB.info())
#%%
Wells55_GWSI_MasterDB = Wells55_GWSI_MasterDB.groupby(['Combo_ID']).first()
print(Wells55_GWSI_MasterDB.info())

# %%
Wells55_GWSI_MasterDB.to_csv('../MergedData/Output_files/Master_ADWR_database_nocancelled.csv')
Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database_nocancelled.shp')

# %%
#  - Only water wells 
Wells55_GWSI_MasterDB = pd.merge(gwsi_gdf, wells55_water, suffixes=['_gwsi','_wells55'], how="outer", 
                                          on=['OBJECTID','Combo_ID',"REGISTRY_I", 'WELL_DEPTH', 'geometry', 'Original_DB'])

print(Wells55_GWSI_MasterDB.info())

# %%
Wells55_GWSI_MasterDB = Wells55_GWSI_MasterDB.groupby(['Combo_ID']).first()
print(Wells55_GWSI_MasterDB.info())

# %%
Wells55_GWSI_MasterDB.to_csv('../MergedData/Output_files/Master_ADWR_database_water.csv')
Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database_water.shp')
# %% Follow up change to add year to the end
filename_mdb_w = 'Master_ADWR_database_water.shp'
filepath = os.path.join(outputpath, filename_mdb_w)
print(filepath)

masterdb_water = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb_water.info())
