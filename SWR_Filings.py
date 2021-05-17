# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

# %%
app_registry = 'ADWR_SW_APPL_REGRY.csv'
CD_Actions = 'ADWR_SW_CD_ACTIONS.csv'
Filestats = 'ADWR_SW_CD_FILESTATS.csv'
swsheds = 'ADWR_SW_CD_SWSHEDS.csv'
wildareas = 'ADWR_SW_CD_WILDAREAS.csv'
wsheds = 'ADWR_SW_CD_WSHEDS.csv'
conveyances = 'ADWR_SW_CONVEYANCES.csv'
locations = 'ADWR_SW_LOCATIONS.csv'
reservoirs = 'ADWR_SW_RESERVOIRS.csv'
tscases = 'ADWR_SW_TSCASES.csv'

filenames = [app_registry, CD_Actions, Filestats, swsheds, \
            wildareas, wsheds, conveyances, locations, reservoirs, tscases]

file_in = []

datapath = '/Users/danielletadych/Documents/PhD_Materials/Data/SurfaceWater/SWR'

x = 0
# %%
for item in filenames:
    filepath = os.path.join(datapath, item)
    print(filepath)
    item = pd.read_csv(filepath)
    print(item)
#%%
    pd.read_csv(filepath)
    pd.options.display.float_format = '{:.2f}'.format


# %%
datapath = '/Users/danielletadych/Documents/PhD_Materials/Data/SurfaceWater/SWR'
filepath = os.path.join(datapath, filename)
print(filepath)

wl_data2 = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wl_data2.info())



# %%