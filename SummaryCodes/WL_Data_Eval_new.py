# Reading in the water level data form the non-transducer wells

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# %%
# Read in the the water  level file
GWSI_folder = 'GWSI_04142020' #GWSI folder name
file_name = 'GWSI_WW_LEVELS.xlsx'
filepath=os.path.join(GWSI_folder, 'Data_Tables', file_name)
print(filepath)

wl_data = pd.read_excel(filepath)
print(wl_data.info())

# Rename the columns to something shorter
wl_data=wl_data.rename(columns={"WLWA_MEASUREMENT_DATE": "date",
                   "WLWA_SITE_WELL_SITE_ID": "wellid",
                   "WLWA_DEPTH_TO_WATER": "depth"}, errors="raise")
# %%
# read in the transducer data file
file_name = 'GWSI_Transducer_Levels.txt'
filepath=os.path.join(GWSI_folder, 'Data_Tables', file_name)
print(filepath)

trans_data=pd.read_table(filepath, sep = ',',  index_col=False, parse_dates=['datetime'],
        names=['wellid', 'unknown', 'datetime', 'depth', 'agency', \
            'method', 'remarks', 'temperature', 'unknown1',  \
            'unknown2', 'notes']
        )
trans_data['date'] = trans_data['datetime'].dt.date

# %%
# Get lists of well IDS
#for the water level file
id_list=wl_data['wellid'].unique()
id_list.sort()
print('Water Level file:', len(id_list), ' unique wells with measurements')

#for the transducer file
id_listT=trans_data['wellid'].unique()
id_listT.sort()
print('Transducer file:', len(id_listT), ' unique wells with measurements')

#combined
id_listC=np.union1d(id_list, id_listT)
print('Combined:', len(id_listC), ' unique wells with measurements')

# %%
#  Summarize  the water level data
wl_summary=pd.pivot_table(wl_data, index='wellid',
                     values=['date', 'depth'],
                     aggfunc={'date': [np.min, np.max],
                             'depth': [np.min, np.max, len]})
wl_summary['minyear'] = pd.DatetimeIndex(summary['date','amin']).year
wl_summary['maxyear'] = pd.DatetimeIndex(summary['date','amax']).year

#condense the column names
wl_summary.columns = [c[0] + '.' +c[1] for c in wl_summary.columns]

# %%
# Summarize the transducer data
t_summary=pd.pivot_table(trans_data, index='wellid', values=['datetime', 'depth'],
                     aggfunc={'datetime': [np.min, np.max],
                               'depth': [np.min, np.max, len]})
t_summary['minyear'] = pd.DatetimeIndex(t_summary['datetime','amin']).year
t_summary['maxyear'] = pd.DatetimeIndex(t_summary['datetime','amax']).year

# aggregated these values to daily
t_dailyS = trans_data.groupby(['date','wellid'])['depth'].mean()
t_daily = t_dailyS.to_frame(name='depth')
t_daily.reset_index(inplace=True)


# Summarize the  daily values
tdaily_summary=pd.pivot_table(t_daily, index='wellid', values=['date', 'depth'],
                     aggfunc={'date': [np.min, np.max],
                               'depth': [np.min, np.max, len]})
tdaily_summary['minyear'] = pd.DatetimeIndex(tdaily_summary['date','amin']).year
tdaily_summary['maxyear'] = pd.DatetimeIndex(tdaily_summary['date','amax']).year

#condense the column names
tdaily_summary.columns = [c[0] + '.' +c[1] for c in tdaily_summary.columns]


# %%
# Join together the two dataframes and write them out
summary_join = wl_summary.join(tdaily_summary, how='outer', lsuffix='', rsuffix='trans')

# print this out to a csv
summary_join.to_csv('Well_Data_Summary.csv', na_rep='Nan')
