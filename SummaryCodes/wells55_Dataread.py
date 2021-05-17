# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import geopandas as gp

#%%
# Read in Wells 55 Data
# This is a file with water levels from ADWR which has been joined with another ADWR file with variables
# so that basinid is also a variable
filename = 'Wells55.csv'
datapath = '/Users/danielletadych/Documents/PhD_Materials/Data/Groundwater'
outputpath = '../MergedData/Output_files'
filepath = os.path.join(datapath, filename)
print(filepath)

wells55 = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wells55.info())

#%% 
# Adding the year
filename = 'Wells55_YEAR.csv'
filepath = os.path.join(datapath, filename)
wells55_year = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wells55_year.head())

# %%
wells55_year['INSTALL_YEAR']
# %%
wells55['INSTALL_YEAR'] = wells55_year['INSTALL_YEAR']
# %%
wells55.head()
# %%
# Reading in the shapefiles
welldir = "/Users/danielletadych/Documents/PhD_Materials/Data/Maps/ADWR/Well_Registry__Wells55_-shp"
regdir = "/Users/danielletadych/Documents/PhD_Materials/PhD_Materials/Mapping/DataDivisions/Dissovled"
unregdir = "/Users/danielletadych/Documents/PhD_Materials/PhD_Materials/Mapping/DataDivisions"
wellfilename = "Well_Registry__Wells55_.shp"
Wellfp = os.path.join(welldir, wellfilename)
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
# %%
# --- Overlaying Regions ---
georegions = [AIH, ID, Statereg, NWu, SWu, SEu, CEu]
wellgeoregions = wells55shape
gp.overlay(wellgeoregions, ID, how='union')

# %%
for i in georegions:
#    print(i.info())
    gp.overlay(wellgeoregions, i, how='union')
    print("Overlay finished:", i)
    wellgeoregions.info()

# %%
# Save file
wellgeoregions.to_file(outputpath + "wells55georegions.shp")
# %%
# Now let's make a pivot table to get some statistics
wellgeoregions['INSTALL_YEAR'] = wells55_year['INSTALL_YEAR']
# %%
welldepth = pd.pivot_table(wellgeoregions, index=['GEO_Region'], values='WELL_DEPTH', aggfunc=['mean'])
# %%
newgeo = os.path.join('../MergedData/', "Output_fileswells55georegions.shp")
print(newgeo)
newgeo = gp.read_file(newgeo)
# %%
