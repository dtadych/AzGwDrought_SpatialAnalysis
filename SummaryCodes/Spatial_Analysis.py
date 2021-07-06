# The purpose of this script is to create a code to spatially analyze all the wells in the combined database based on management. 
# Written by Danielle Tadych
# Goals:
# - create columns with the different management regions
#       1. AMA/INA
#       2. Irrigation District
#       3. AI Homelands
#       4. Unregulated

# Potential workflow
# - import master database and shape files
# - Create columns of the different management options
# - Create an if statement of if a well falls within a certain region, then it can equal the name of the shape it falls under

# %%
import os
from geopandas.tools.sjoin import sjoin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
import earthpy as et

# %%
# Load in the master database

# Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database.shp')

filename = 'Master_ADWR_database.shp'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
filepath = os.path.join(outputpath, filename)
print(filepath)

masterdb = gp.read_file(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(masterdb.info())
# %%
# Reading in the shapefiles
AIHdir = "/Users/danielletadych/Documents/PhD_Materials/PhD_Materials/Mapping/"
shpdir = "/Users/danielletadych/Documents/PhD_Materials/PhD_Materials/Mapping/DataDivisions"
# Regulated
AIH_fn = "AZ_AIHomelands.shp"
Irrig_Dist_fn = "Irrigation_District_FixedGeometries.shp"
StateReg_fn = "State_Regulated_NonAI.shp"

AIHfp = os.path.join(AIHdir, AIH_fn)
IDfp = os.path.join(shpdir, Irrig_Dist_fn)
State = os.path.join(shpdir, StateReg_fn)

AIH = gp.read_file(AIHfp)
ID = gp.read_file(IDfp)
Statereg = gp.read_file(State)
# %%
AIH.head()
# %%
Statereg.info()
# %%
ID.head()
# %%
# ---- Now joining the spatial dataframes to the master database ----
# Using Spatial join based on this webpage: https://geopandas.org/gallery/spatial_joins.html

ID_short = ID[["OBJECTID_1", "IRR_NAME", "SHAPEAREA", "SHAPELEN", "geometry"]]

# %%
georegdb = gpd.sjoin(masterdb, ID_short, how="left")
georegdb = georegdb.drop(['index_right', 'OBJECTID_1', 'SHAPEAREA', 'SHAPELEN'], axis=1)
georegdb.head()
# %% Need to make a list of columns that are not NaN in IRR_NAME
Irrgeodb = georegdb[georegdb['IRR_NAME'].notnull()]
IrrRegIDlist = list(Irrgeodb["REGISTRY_I"])
# %% Read in the annual timeseries database
filename = 'Wells55_GWSI_WLTS_DB_annual.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)

annual_db = pd.read_csv(filepath, header=1, index_col=0)
pd.options.display.float_format = '{:.2f}'.format
annual_db.head()
# %% Making a sub annual timeseries dataframe based soley on values from our list
for i in IrrRegIDlist:
    IrrTS = annual_db.loc[[i]].append()

IrrTS.info()
# %% Get the average for all the columns
df.mean(axis = 0)