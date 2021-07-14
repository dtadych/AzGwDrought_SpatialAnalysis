# The purpose of this code is to create a single shapefile with all the georgions for the Spatial Analysis
# Written by Danielle Tadych

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

def readshapefile(file_string):
    """
    This function will read in a shapefile so I can put it into a loop
    """
    filename = file_string + ".shp"
    filepath = os.path.join(datapath, filename)
    print(filepath)
    gp.read_file(filepath)

# %% Setting up the file environment
datapath = '../MergedData/Shapefiles/'
Shapefiles = ["AZ_AIHomelands","County_NonAI","Irrigation_District_FixedGeometries","State_Regulated_NonAI","Unregulated_GWSBs",]

# %%
#Shapefiles = {"AIH" : "AZ_AIHomelands",
#                "AZCounties":"AZ_counties",
#                "IrrDist":"Irrigation_District_FixedGeometries",
#                "State":"State_Regulated_NonAI",
#                "Unreg":"Unreg_NonAI"}
#%%
li = []
for x,y in Shapefiles.items():
    filename = y + ".shp"
    filepath = os.path.join(datapath, filename)
    print(filepath)
    df = gp.read_file(filepath)
    li.append(df)


# %%
AIH = readshapefile("AZ_AIHomelands")
AZCounties = readshapefile("AZ_Counties")
Irrig_Dist = readshapefile("Irrigation_District_FixedGeometries")
State = readshapefile("State_Regulated_NonAI")
Unreg = readshapefile("Unreg_NonAI")
# %%
# Reading in the shapefiles
filename = Shapefiles[0] + ".shp"
filepath = os.path.join(datapath, filename)
print(filepath)
AIH = gp.read_file(filepath)

filename = Shapefiles[1] + ".shp"
filepath = os.path.join(datapath, filename)
print(filepath)
AZCounties = gp.read_file(filepath)

filename = Shapefiles[2] + ".shp"
filepath = os.path.join(datapath, filename)
print(filepath)
IrrDist = gp.read_file(filepath)

filename = Shapefiles[3] + ".shp"
filepath = os.path.join(datapath, filename)
print(filepath)
State = gp.read_file(filepath)

filename = Shapefiles[4] + ".shp"
filepath = os.path.join(datapath, filename)
print(filepath)
Unreg = gp.read_file(filepath)
# %% Need to make a new dataframe that combines AIH and AZ Counties
County_terr = AIH.merge(AZCounties, how='outer', on=['geometry', 'NAME'])
fig, ax = plt.subplots()
County_terr.plot(ax = ax)
County_terr['NAME'].unique()
# %%
County_terr = County_terr[['NAME', 'geometry']]
County_terr
fig, ax = plt.subplots()
County_terr.plot(ax = ax)
# %%
#Shortnames = [AIH, AZCounties, IrrDist, State, Unreg]
Shortnames = {"AIH" : AIH,
#                "AZCounties":AZCounties,
                "IrrDist":IrrDist,
                "StateReg":State,
                "Unreg":Unreg}
# %%
# Adding columns to each shapefile for Georegion category (one of the short names) and also the specific name of each section (e.g. "TUCSON AMA"; GEO_Reg_Name)
for x, y in Shortnames.items():
    y['GEO_Region_Cat'] = x
    y['GEO_Region_Name'] = np.nan
    Shortnames[x].append(y)
    print(Shortnames.items())
#    print(y.head())
# %% Applying those to the geodataframes outside of the dictionary
AIH = Shortnames["AIH"]
#AZCounties = Shortnames["AZCounties"]
IrrDist = Shortnames["IrrDist"]
State = Shortnames["StateReg"]
Unreg = Shortnames["Unreg"]
# %% Now making the specific names for each of the columns
AIH["GEO_Region_Name"] = AIH["NAME"]
#AZCounties["GEO_Region_Name"] = AZCounties["NAME"]
IrrDist["GEO_Region_Name"] = IrrDist["LONG_NAME"]
State["GEO_Region_Name"] = State["BASIN_NAME"]
# Note - "NAME" in Unregulated geodataframe is county name and SUBBASIN_1 is full subbasin name
Unreg["GEO_Region_Name"] = Unreg["SUBBASIN_1"]
# %% Checking to see if it's plotting, if they overlap, and if the columns
#Shortnames = [AIH, IrrDist, State, Unreg]
newlist = []
for x, y in Shortnames.items():
    y = y[['geometry', 'GEO_Region_Cat', 'GEO_Region_Name']]
    print(i.info(),y['GEO_Region_Cat'].unique())
    newlist.append(y)
    fig, ax = plt.subplots()
    y.plot(ax=ax,label=x)
    y.to_file(x+".shp")

# %% Anything I do to combine these from here isn't fucking working so I'm going to union in qgis and then go from there
# QGIS worked!  Reading in the new shapefile
filename = "GEOREGIONS.shp"
filepath = os.path.join(datapath, filename)
print(filepath)
GEOREG = gp.read_file(filepath)
# %%
GEOREG.rename(columns={'GEO_Region':'GEO_REG_CAT','GEO_Regi_1':'GEO_REG_NAME'}, inplace=True)
# %%
AZCounties.info()
fig, ax = plt.subplots()
AZCounties.plot(ax = ax)
# %%
GEOREG['Counties_Territories'] = County_terr['NAME']
GEOREG.info()

# %%
fig, ax = plt.subplots()
GEOREG.plot(ax = ax)
#%%
newlist
# %%
StartShape = newlist[0]
fig, ax = plt.subplots()
StartShape.plot(ax = ax)
StartShape.info()

#%%
Nextshape = newlist[2]
Combinedshape = Nextshape.merge(StartShape, how='outer', on=['geometry', 'GEO_Region_Cat', 'GEO_Region_Name'])
fig, ax = plt.subplots()
Combinedshape.plot(ax = ax)
Combinedshape.info()
#%%
Combinedshape = newlist[2].merge(Combinedshape, how='outer', on=['geometry', 'GEO_Region_Cat', 'GEO_Region_Name'])
fig, ax = plt.subplots()
StartShape.plot(ax = ax)
Combinedshape.info()
#%%
for i in newlist:
    print(i.info())
    Combinedshape = i.merge(StartShape, how='outer', on=['geometry', 'GEO_Region_Cat', 'GEO_Region_Name'])
    #print(Combinedshape.unique())

Combinedshape.info()
#%%
fig, ax = plt.subplots()
Combinedshape.plot(ax = ax)

#%%
#Wells55_GWSI_MasterDB = wells55_gdf.merge(gwsi_gdf, suffixes=['_wells55','_gwsi'], how="outer", left_on="REGISTRY_I", right_on="REG_ID")
Wells55_GWSI_MasterDB = wells55_gdf.merge(gwsi_gdf, suffixes=['_wells55','_gwsi'], how="outer", on=['']
                                          left_on=["REGISTRY_I", 'WELLTYPE', 'WELL_DEPTH', 'geometry', 'Original_DB'],
                                          right_on=["REG_ID", 'WELL_TYPE', 'WELL_DEPTH', 'geometry', 'Original_DB'])
print(Wells55_GWSI_MasterDB.info())

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
Wells55_GWSI_MasterDB.to_file("Master_ADWR_Database.shp")
Wells55_GWSI_MasterDB.to_csv('../MergedData/Output_files/Master_ADWR_database.csv')
# %%
Wells55_GWSI_MasterDB.to_file('../MergedData/Output_files/Master_ADWR_database.shp')
