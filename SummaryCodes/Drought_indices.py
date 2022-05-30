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
nclimdata = nclimdata.rename(columns = {'   PDSI':'PDSI', '   PHDI':'PHDI'})

# %%
pdsi = nclimdata[['date','PDSI','PHDI']]
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
yearly_pdsi = pd.pivot_table(pdsi, index=["In_year"], values=["PDSI", 'PHDI'], dropna=False, aggfunc=np.mean)
yearly_pdsi
# %%
yearly_pdsi.plot()

#%%
drought_color = '#ffa6b8'
wet_color = '#b8d3f2'

# %%
value = 1
yearly_pdsi['wet'] = value
yearly_pdsi['dry'] = -value
yearly_pdsi
#  PDSI
ds = yearly_pdsi
minyear=1975
maxyear=2020
name = "Average PDSI and PHDI for AZ from " + str(minyear) + " to " + str(maxyear)
min_y = -6
max_y = 6
fsize = 12

fig, ax = plt.subplots(figsize = (9,5))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
# ax.plot(ds['CAP'], label='CAP', color=c_2)
ax.plot(ds['PDSI']
        , label='PDSI'
        , color='#e77a47'
        , lw=2
        ) 
ax.plot(ds['PHDI'], label='PHDI'
        # , color=''
        , lw=1
        ) 
# ax.plot(ds['wet'],label='wet',color='black',zorder = 5)
ax.plot(ds['dry'],label='Cutoff Value',color='black', zorder=5)
a = 1975
b = 1977.5
c = 1980.5
d = 1981.5
e = 1988.5
f = 1990.5
g = 1995.5
h = 1997.5
i = 1998.5
j = 2004.5
k = 2005.5
l = 2009.5
m = 2010.5
n = 2018.5
plt.axvspan(a, b, color=drought_color, alpha=0.5, lw=0, label="Drought")
plt.axvspan(c, d, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(e, f, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(g, h, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(i, j, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(k, l, color=drought_color, alpha=0.5, lw=0)
plt.axvspan(m, n, color=drought_color, alpha=0.5, lw=0)

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
ax.minorticks_on()
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_title(name, fontsize=14)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Index Values',fontsize=fsize)
ax.legend(loc = [1.04, 0.40], fontsize = fsize)
fig.set_dpi(600.0)


# %%
value = 1
print("Drought is considered > -", value)
yearly_pdsi['wet'] = value
yearly_pdsi['dry'] = -value
yearly_pdsi

drought = yearly_pdsi[yearly_pdsi['PHDI']<=-value]
wet = yearly_pdsi[yearly_pdsi['PHDI']>=value]
drought = drought[drought.index >= 1975]
wet = wet[wet.index >= 1975]

print()
print("Drought Year Info:")
print(drought['PHDI'].describe())
print()
print("Wet Year Info:")
print(wet['PHDI'].describe())

#  PHDI
ds = yearly_pdsi
minyear=1975
maxyear=2020
name = "Average PHDI for AZ from " + str(minyear) + " to " + str(maxyear)
min_y = -6
max_y = 6
fsize = 12

fig, ax = plt.subplots(figsize = (9,5))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
# ax.plot(ds['CAP'], label='CAP', color=c_2)
# ax.plot(ds['PDSI'], '-o'
#         , label='PDSI'
#         , color='grey'
#         , lw=1
#         ) 
ax.plot(ds['PHDI'], '-',label='PHDI'
        , color='grey'
        , lw=1
        ) 
ax.plot(ds['wet'],color='black',zorder = 5)
ax.plot(ds['dry'],color='black', zorder=5)
ax.plot(drought['PHDI'],'o',label='dry',color='red', zorder=5)
ax.plot(wet['PHDI'],'o',label='wet',color='blue', zorder=5)

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
ax.minorticks_on()
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_title(name, fontsize=14)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Index Values',fontsize=fsize)
# ax.legend(loc = [1.04, 0.40], fontsize = fsize)
fig.set_dpi(600.0)

#%%
# value = 1.5
print("Drought is considered > -", value)
yearly_pdsi['wet'] = value
yearly_pdsi['dry'] = -value
yearly_pdsi

drought = yearly_pdsi[yearly_pdsi['PDSI']<=-value]
wet = yearly_pdsi[yearly_pdsi['PDSI']>=value]
drought = drought[drought.index >= 1975]
wet = wet[wet.index >= 1975]

print()
print("Drought Year Info:")
print(drought['PDSI'].describe())
print()
print("Wet Year Info:")
print(wet['PDSI'].describe())

#  PDSI
ds = yearly_pdsi
minyear=1975
maxyear=2020
name = "Average PDSI for AZ from " + str(minyear) + " to " + str(maxyear)
min_y = -6
max_y = 6
fsize = 12

fig, ax = plt.subplots(figsize = (9,5))
#ax.plot(ds[1.0], label='Reservation', color=c_1)
# ax.plot(ds['CAP'], label='CAP', color=c_2)
# ax.plot(ds['PDSI'], '-o'
#         , label='PDSI'
#         , color='grey'
#         , lw=1
#         ) 
ax.plot(ds['PDSI'], '-',label='PDSI'
        , color='grey'
        , lw=1
        ) 
ax.plot(ds['wet'],color='black',zorder = 5)
ax.plot(ds['dry'],color='black', zorder=5)
ax.plot(drought['PDSI'],'o',label='dry',color='red', zorder=5)
ax.plot(wet['PDSI'],'o',label='wet',color='blue', zorder=5)

ax.set_xlim(minyear,maxyear)
ax.set_ylim(min_y,max_y)
ax.minorticks_on()
ax.grid(visible=True,which='major')
ax.grid(which='minor',color='#EEEEEE', lw=0.8)
ax.set_title(name, fontsize=14)
ax.set_xlabel('Year', fontsize=fsize)
ax.set_ylabel('Index Values',fontsize=fsize)
# ax.legend(loc = [1.04, 0.40], fontsize = fsize)
fig.set_dpi(600.0)

# %%
analysis_period = yearly_pdsi[yearly_pdsi.index>=1975]
del analysis_period['wet']
del analysis_period['dry']

analysis_period.to_csv('../MergedData/Output_files/Yearly_DroughtIndices.csv')

# %%
analysis_period
# %%
drought_phdi = analysis_period[analysis_period['PHDI']<=-1]
drought_pdsi = analysis_period[analysis_period['PDSI']<=-1]
print(drought_pdsi)
# %%
print(drought_phdi)
# %%
df = analysis_period
value = 1
df['PHDI_status'] = 'Normal'
df.loc[df['PHDI'] <= -value, 'PHDI_status'] = 'Drought' 
df.loc[df['PHDI'] >= value, 'PHDI_status'] = 'Wet' 
df['PDSI_status'] = 'Normal'
df.loc[df['PDSI'] <= -value, 'PDSI_status'] = 'Drought' 
df.loc[df['PDSI'] >= value, 'PDSI_status'] = 'Wet' 

df

analysis_period.to_csv('../MergedData/Output_files/YearlyDrought_'+str(value)+'.csv')

# %%
analysis_period[['PHDI_status','PDSI_status']].describe()
# %%
print('Drought info: ', analysis_period[analysis_period['PDSI_status']=='Drought'].describe())

# %%
df
# %%
print()