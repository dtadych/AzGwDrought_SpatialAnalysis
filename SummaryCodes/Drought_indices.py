# ------ Drought Indice Analysis --------
# Written by Danielle Tadych, May 2022

# The purpose of this code is to pick out drought periods 
# based on PDSI

# The dataset needed is nClimDiv text file from the GHCN.  This data in particular is
# averaged for the state of Arizona


# %% Load the packages
from cProfile import label
from operator import ge
from optparse import Values
import os
from geopandas.tools.sjoin import sjoin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import datetime as dt
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
#import earthpy as et
import scipy.stats as sp

# Assign Data paths
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
shapepath = '../MergedData/Shapefiles/Final_Georegions/'

# %% Read in the file
filename = 'nClimDiv_AZ_GHCN.txt'
filepath = os.path.join(datapath, filename)
print(filepath)

#%%
nclimdata = pd.read_csv(filepath 
                        #   ,parse_dates=['INSTALLED']
                          )
nclimdata
# %%
nclimdata.describe()
# %%
nclimdata['date'] = pd.to_datetime(nclimdata['YearMonth'], format='%Y%m', errors='coerce').dropna()
nclimdata

# %%
nclimdata = nclimdata.rename(columns = {'   PDSI':'PDSI'})

# %%
pdsi = nclimdata[['date','PDSI']]
pdsi
# %%
pdsi.describe()
# %%
pdsi = pdsi.set_index('date')
pdsi
# %%
pdsi.plot()

# %%
pdsi = pdsi.reset_index()
# %%
pdsi['In_year'] = pdsi['date'].dt.year
pdsi

# %%
yearly_pdsi = pd.pivot_table(pdsi, index=["In_year"], values=["PDSI"], dropna=False, aggfunc=np.mean)
yearly_pdsi
# %%
yearly_pdsi.plot()
# %%
