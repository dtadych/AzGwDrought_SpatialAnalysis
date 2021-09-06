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

# %% Anything I do to combine these from here isn't working so I'm going to union in qgis and then go from there
# QGIS worked!  Reading in the new shapefile
filename = "GEOREGIONS.shp"
filepath = os.path.join(datapath, filename)
print(filepath)
GEOREG = gp.read_file(filepath)
# %%
GEOREG.rename(columns={'GEO_Region':'GEO_REG_CAT','GEO_Regi_1':'GEO_REG_NAME'}, inplace=True)
# %% Adding third column of counties and territories for large statistics
County_terr.rename(columns={'NAME':'Counties_Territories'}, inplace=True)
County_terr.info()

# %%
georegtest = gp.sjoin(GEOREG, County_terr, how='left')
georegtest.info()
# %% Getting rid of the unnecessary columns
GEOREG = GEOREG[['GEO_REG_CAT','GEO_REG_NAME','geometry','Counties_Territories']]
#%%
GEOREG['GEO_REG_CAT'].unique()
# %%
GEOREG.info()
# %%
fig, ax = plt.subplots()
GEOREG.plot(ax = ax)
#%%
GEOREG.to_file('../MergedData/Output_files/Georegions_3col.shp')

# %%
GEOREG[(GEOREG.GEO_REG_CAT == "StateReg")].head()
# %%
