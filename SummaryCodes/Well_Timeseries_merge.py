# The purpose of this script is to make multiple timeseries databases using data from the GWSI and Wells55 databases
# Written by Danielle Tadych
# Goals:
# - Create columns in each respective database specifying its origin
# - Find column they have in common
# - Merge based on that column
# - Make GWSI wells the overriding database and fill in the gaps with Wells55

# - Make 3 Timeseries databases: WL, Water Elevation, and Pumping
#     * Note: Pumping might not be available in Wells55
#             Would need to potentially multiply pumping amounts with how long it has been installed
#             Check with Laura
# - Make the columns well ID's and the rows dates
# - for Wells55 water level and elevation, need to make rows the install date and columns REGISTRY_I
# - Merge based on well ID's

# Status as of 7/5/2021:
# - Made water level (depth to water below land surface in feet) timeseries databases
# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from pandas.tseries.offsets import BYearBegin
import seaborn as sns
import geopandas as gp

# %% 
# ----- Import the Data and Shapefiles with Geometries -----
# Read in Wells 55 Data
# This is a file with water levels from ADWR which has been joined with another ADWR file with variables
filename = 'Wells55.csv'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files/'
filepath = os.path.join(datapath, filename)
print(filepath)

wells55 = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wells55.info())

# Read in pump_wl 
# This is a combined file with pump data & depth to water
filename = 'Pump_wl.csv'
filepath = os.path.join(datapath, filename)
print(filepath)

pump_wl = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(pump_wl.info())

# Read in GWSI collated water level data
filename = 'wl_data2.csv'
filepath = os.path.join(datapath, filename)
print(filepath)

wl_data2 = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wl_data2.info())

#%%
# Read in GWSI collated pumping data
filename = 'Pump_Data_Full.csv'
filepath = os.path.join(datapath, filename)
print(filepath)

pump_data_all = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wl_data2.info())

#%% ---- Making dataframes with Date as the index ---
# Make dataframes with Columns for GWSI and Wells55, respectively
# Following this method: https://stackoverflow.com/questions/32215024/merging-time-series-data-by-timestamp-using-numpy-pandas
# Confirmed through the variables list that "depth" in wldata2 and "WATER_LEVE" are both depth to water below land surface in feet

gwsi_wl = wl_data2[["date","SITE_WELL_REG_ID","depth"]].copy()
gwsi_wl.info()

wells55_wl = wells55[["INSTALLED", "REGISTRY_I", "WATER_LEVE"]].copy()
wells55_wl.info()

# %%
# Need to add an original database column
wells55_wl["Original_DB"] = 'Wells55'
gwsi_wl["Original_DB"] = 'GWSI'
wells55_wl.head()

# %%
wells55_wl.rename(columns = {'INSTALLED':'date','WATER_LEVE':'depth'}, inplace=True)
gwsi_wl.rename(columns={'SITE_WELL_REG_ID':'REGISTRY_I'}, inplace=True)

#%%
#combo = gwsi_wl.join(wells55_wl, how='outer')
#combo

combo = wells55_wl.merge(gwsi_wl, suffixes=['_wells55','_gwsi'], how="outer" 
                                          ,on=["REGISTRY_I", 'date', 'Original_DB', 'depth']
                                          )
combo.info()

# %% Set date as index to conver to datetime
combo.set_index("date", inplace=True)
combo.index = pd.to_datetime(combo.index)
combo.info()
# %%
WL_TS_DB = pd.pivot_table(combo, index=["REGISTRY_I"], columns="date", values="depth")
# %%
WL_TS_DB.head()

# %%
# Export data into a csv
WL_TS_DB.to_csv(outputpath + 'Wells55_GWSI_WLTS_DB.csv')

# %% --- Summarizing the data by date ---
# Extract the year from the date column and create a new column year
combo['year'] = pd.DatetimeIndex(combo.index).year
combo.head()

# %%
WL_TS_DB_year = pd.pivot_table(combo, index=["REGISTRY_I"], columns=["year"], values=["depth"], dropna=False, aggfunc=np.mean)
# %%
print(WL_TS_DB_year.iloc[:,115])
# %%
print(WL_TS_DB_year.iloc[:,155])

# %%
WL_TS_DB_1980 = pd.DataFrame(WL_TS_DB_year.iloc[:,115])
WL_TS_DB_2020 = pd.DataFrame(WL_TS_DB_year.iloc[:,155])
WL_TS_DB_1980.index.name = None
WL_TS_DB_2020.index.name = None
WL_TS_DB_1980.columns = ["depth"]
WL_TS_DB_2020.columns = ["depth"]

#%%
WL_TS_DB_year.index.name = None
WL_TS_DB_year.head()
# %% Exporting data
WL_TS_DB_1980.to_csv(outputpath + 'comboDB_WL_1980.csv')
WL_TS_DB_2020.to_csv(outputpath + 'comboDB_WL_2020.csv')
# %%  Seeing if things work
fig, ax = plt.subplots()
ax.plot(WL_TS_DB_year.iloc[:,155])
ax.set(title='WL in 1980', xlabel='Registry ID', ylabel='Water Level (feet)')
ax.grid()
plt.show
# %%
WL_TS_DB_year.info()
# %%
# Summarize by Monthly now
combo_monthly2 = combo
combo_monthly2.reset_index(inplace=True)
combo_monthly2.date = combo.date.dt.strftime('%Y-%m')
combo_monthly2.info()
# %%
WL_TS_DB_month = pd.pivot_table(combo_monthly2, index=["REGISTRY_I"], columns=["date"], values=["depth"], dropna=False, aggfunc='mean')
WL_TS_DB_month.info()
WL_TS_DB_month.index.name = None
WL_TS_DB_month.head()
# %%
# Export both yearly summary data and monthly into csv
WL_TS_DB_year.to_csv(outputpath + 'Wells55_GWSI_WLTS_DB_annual.csv')
# %%
WL_TS_DB_month.to_csv(outputpath + 'Wells55_GWSI_WLTS_DB_monthly.csv')
# %%
# Making a for loop for years
# Going to re-read in the combined database so we can get rid of that first row
filename = 'Wells55_GWSI_WLTS_DB_annual.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)

annual_db = pd.read_csv(filepath, header=1, index_col=0)
pd.options.display.float_format = '{:.2f}'.format
annual_db.head()

# %%
columns = annual_db.columns.values.tolist()
print(columns)

# %%
for i in columns:
    WL_TS = 'WL_TS_DB_' + i
    print(WL_TS)
    f = pd.DataFrame(annual_db[i])
    f.index.name = None
    f.columns = ["depth"]
    print(f.describe())
#    print(outputpath + 'YearlyTS/' + WL_TS + '.csv')
    f.to_csv(outputpath + 'YearlyTS/' + WL_TS + '.csv')

# %%
# Making a for loop for months
# Going to re-read in the combined database so we can get rid of that first row
filename = 'Wells55_GWSI_WLTS_DB_monthly.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)

monthly_db = pd.read_csv(filepath, header=1, index_col=0)
pd.options.display.float_format = '{:.2f}'.format
monthly_db.head()

# %%
columns = monthly_db.columns.values.tolist()
print(columns)

# %% Testing before the for loop
i = '2007-05'
WL_TS = 'WL_TS_DB_' + i
print(WL_TS)
f = pd.DataFrame(monthly_db[i])
f.index.name = None
f.columns = ["depth"]
print(f.describe())
print(outputpath + 'MonthlyTS/' + WL_TS + '.csv')


# %%
for i in columns:
    WL_TS = 'WL_TS_DB_' + i
    print(WL_TS)
    f = pd.DataFrame(monthly_db[i])
    f.index.name = None
    f.columns = ["depth"]
    print(f.describe())
#    print(outputpath + 'MonthlyTS/' + WL_TS + '.csv')
    f.to_csv(outputpath + 'MonthlyTS/' + WL_TS + '.csv')

# %%
# ---- Summarizing the data by number of measurements ----

LEN_TS_DB_year = pd.pivot_table(combo, index=["REGISTRY_I"], columns=["year"], values=["depth"], dropna=False, aggfunc=len)
LEN_TS_DB_year.index.name = None
LEN_TS_DB_year.head()

# %%
LEN_TS_DB_month = pd.pivot_table(combo_monthly2, index=["REGISTRY_I"], columns=["date"], values=["depth"], dropna=False, aggfunc=len)
LEN_TS_DB_month.index.name = None
LEN_TS_DB_month.head()
# %%
# Export both yearly summary data and monthly into csv
LEN_TS_DB_year.to_csv(outputpath + 'Wells55_GWSI_LEN_TS_DB_annual.csv')
# %%
LEN_TS_DB_month.to_csv(outputpath + 'Wells55_GWSI_LEN_TS_DB_monthly.csv')
# %%
# Making a for loop for years
# Going to re-read in the combined database so we can get rid of that first row
filename = 'Wells55_GWSI_LEN_TS_DB_annual.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)

annual_db = pd.read_csv(filepath, header=1, index_col=0)
pd.options.display.float_format = '{:.2f}'.format
annual_db.describe()

# %%
columns = annual_db.columns.values.tolist()
print(columns)

# %%
for i in columns:
    LEN_TS = 'LEN_TS_DB_' + i
    print(LEN_TS)
    f = pd.DataFrame(annual_db[i])
    f.index.name = None
    f.columns = ["depth"]
    print(f.describe())
#    print(outputpath + 'YearlyTS/' + LEN_TS + '.csv')
    f.to_csv(outputpath + 'YearlyTS/' + LEN_TS + '.csv')

# %%
# Making a for loop for months
# Going to re-read in the combined database so we can get rid of that first row
filename = 'Wells55_GWSI_LEN_TS_DB_monthly.csv'
filepath = os.path.join(outputpath, filename)
print(filepath)

monthly_db = pd.read_csv(filepath, header=1, index_col=0)
pd.options.display.float_format = '{:.2f}'.format
monthly_db.head()

# %%
columns = monthly_db.columns.values.tolist()
print(columns)

# %% Testing before the for loop
i = '2007-05'
LEN_TS = 'LEN_TS_DB_' + i
print(LEN_TS)
f = pd.DataFrame(monthly_db[i])
f.index.name = None
f.columns = ["depth"]
print(f.describe())
print(outputpath + 'MonthlyTS/' + LEN_TS + '.csv')


# %%
for i in columns:
    LEN_TS = 'LEN_TS_DB_' + i
    print(LEN_TS)
    f = pd.DataFrame(monthly_db[i])
    f.index.name = None
    f.columns = ["depth"]
    print(f.describe())
#    print(outputpath + 'MonthlyTS/' + LEN_TS + '.csv')
    f.to_csv(outputpath + 'MonthlyTS/' + LEN_TS + '.csv')


# %%
