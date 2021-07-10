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
# %% Setting up the file environment
datapath = '../MergedData/Shapefiles/'
Shapefiles = ["AZ_AIHomelands","AZ_counties","Irrigation_District_FixedGeometries","State_Regulated_NonAI","Unreg_NonAI",]

#%%
for i in Shapefiles:
    filename = i + ".shp"
    filepath = os.path.join(datapath, filename)
    print(filepath)
    f = gp.read_file(filepath)
    os.rename(f, i)

# %%
# Reading in the shapefiles
dissolveddir = "/Users/danielletadych/Documents/PhD_Materials/PhD_Materials/Mapping/DataDivisions/Dissovled"
maindir = "/Users/danielletadych/Documents/PhD_Materials/PhD_Materials/Mapping/DataDivisions"
# Regulated
AIH_fn = "AIH_Dissolved.shp"
Irrig_Dist_fn = "Irrig_Dist_Dissolved.shp"
StateReg_fn = "State_Reg_dissolved.shp"

AIHfp = os.path.join(regdir, AIH_fn)
IDfp = os.path.join(regdir, Irrig_Dist_fn)
State = os.path.join(regdir, StateReg_fn)
#unregulated
NW_ufn = "NW_Unregulated.shp"
SW_ufn = "SW_AZ_Unregulated.shp"
SE_ufn = "SE_AZ_Unreg.shp"
CE_ufn = "CE_AZ_Unregulated.shp"

NWfp = os.path.join(unregdir, NW_ufn)
SWfp = os.path.join(unregdir, SW_ufn)
SEfp = os.path.join(unregdir, SE_ufn)
CEfp = os.path.join(unregdir, CE_ufn)

wells55shape = gp.read_file(Wellfp)
AIH = gp.read_file(AIHfp)
ID = gp.read_file(IDfp)
Statereg = gp.read_file(State)
NWu = gp.read_file(NWfp)
SWu = gp.read_file(SWfp)
SEu = gp.read_file(SEfp)
CEu = gp.read_file(CEfp)