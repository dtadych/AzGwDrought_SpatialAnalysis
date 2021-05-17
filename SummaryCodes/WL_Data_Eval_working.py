# %%
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

#%%
# Read in wl_data2
# This is a file with water levels from ADWR which has been joined with another ADWR file with variables
# so that basinid is also a variable
filename = 'wl_data2.csv'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files'
filepath = os.path.join(datapath, filename)
print(filepath)

wl_data2 = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(wl_data2.info())

#%% 
# Create a pivot table basin and avg depthacross all years 
pivot1 = pd.pivot_table(wl_data2, index='basinid', values='depth', aggfunc=['mean'])
print(pivot1)

#%% 
# Save pivot table 1 to a csv in the specified directory 
pivot1.to_csv(outputpath + 'avgwldepth_bybasin.csv')

#%%
# Create a pivot table wellid and avg depth across all years 
pivot2 = pd.pivot_table(wl_data2, index='wellid', values='depth', aggfunc=['mean'])
print(pivot2)

#%% 
# Save pivot table 2 to a csv in the specified directory 
pivot2.to_csv(outputpath + 'avgwldepth_bywellid.csv')

#%%
# dont think this set is necessary but it is good to know and may need in future
#set index of dataframe by date [] makes it a multiindex if u want
wl_data2.set_index('date')

#%%
# extract the year from the date column and create a new column year
wl_data2['year'] = pd.DatetimeIndex(wl_data2['date']).year

#%%
# Create a pivot table of avg water depth by basin and by year
pivot3 = pd.pivot_table(wl_data2, index=['basinid','year'], values='depth', aggfunc=['mean'])
print(pivot3)

#%%
wl_data_basin = pivot3.unstack(level=1)
print(wl_data_basin)

#%% 
# Save pivot table 3 to a csv in the specified directory 
pivot3.to_csv(outputpath + 'avgwldepth_bybasin&year.csv')

#%%
# Create a pivot table of avg water depth by well id and year
pivot4 = pd.pivot_table(wl_data2, index=['wellid','year'], values='depth', aggfunc=['mean'])
print(pivot4)

#%% 
# Save pivot table 4 to a csv in the specified directory 
pivot4.to_csv(outputpath + 'avgwldepth_bywellid&year.csv')

#%%
# create a dataframe wl_data3 which unstacks pivot table 4 so rows are wellid and columns are each year
wl_data3 = pivot4.unstack(level=1)
print(wl_data3)

# %%
# Counting the number of yearly measurements we have per Well ID
count_wldata_1 = wl_data3.count(axis=1)
count_wldata = pd.DataFrame(count_wldata_1)
count_wldata.columns = ['Measurements']
print(count_wldata)
count_wldata.to_csv(outputpath + 'avgwldepth_wellid&year_Count.csv')


#%%
#Create a variable called "myid" to make locating graphing wells easier
myid=315444111550301

# %%
# take pivot4 and locate all the water levels by year for the given wellid
pivot4.loc[myid]

#%%
# plot the wellid identified by "myid" by wl depth and year 
fig, ax = plt.subplots()
ax.plot(pivot4.loc[myid],label=myid)
ax.set(title='Depth To Water', xlabel='Year', ylabel='Depth(ft)')
ax.legend()
ax.grid()
plt.show

#Save plot as a png with the myid name to the specified directory
type = myid
plt.savefig(outputpath + '{0}.png'.format(type), bbox_inches='tight')

#%%
#list of wells that we want to plot
wl_data2.dropna(subset=['depth'],inplace=True)
basin='AGF'
mylist=wl_data2[wl_data2['basinid']==basin]['wellid'].unique()


# plot the wellid identified by "myid" by wl depth and year 
fig, ax = plt.subplots()
ax.plot(pivot4.loc[mylist[0]],label=mylist[0])
ax.set(title='Depth To Water', xlabel='Year', ylabel='Depth(ft)')


for i in range(1,len(mylist)):
    print(i)
    ax.plot(pivot4.loc[mylist[i]],label=mylist[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.grid()
plt.show

#%%
#unstack pivot3 into a df called wl_data_basin to be able to plot
wl_data_basin = pivot3.unstack(level=0)
print(wl_data_basin)

#%%
# plot the wellid identified by "myid" by wl depth and year 
fig, ax = plt.subplots()
ax.plot(wl_data_basin)
ax.set(title='Depth To Water', xlabel='Year', ylabel='Depth(ft)')
#can plot different x scales
plt.xlim(1920,2012)
#can plot different y scales
plt.ylim(0,500)
ax.legend()
ax.grid()
plt.show

# %%
basin='AGF'
mylist=wl_data2[wl_data2['basinid']==basin].unique()
#%%
# Create a loop where AF Pumped is plotted by basin
fig, ax = plt.subplots()
for i in range(len(mylist)):
    print(i)
    print(mylist[i])
    ax.plot (wl_data_basin[mylist[i]], label=mylist[i])


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
ax.grid()
plt.show
# %%
# Time series of select subbasins

fig, ax = plt.subplots()
ax.plot(pivot3.loc['WIL'],label='WIL, Wilcox (unregulated)')
ax.plot(pivot3.loc['WSR'],label='WSR, Inside Phoenix AMA')
ax.plot(pivot3.loc['USC'],label='USC, Inside Tucson AMA')
ax.plot(pivot3.loc['SSW'],label='SSW, Tohono Oodham Nation')
ax.plot(pivot3.loc['SAF'],label='SAF, Irrigation District')
ax.set(title='Depth To Water by Subbasin', xlabel='Year', ylabel='Depth(ft)')
#ax.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.grid()
plt.show

#Save plot as png to the specified directory
plt.savefig(outputpath + 'WL_SelectBasins.png', bbox_inches='tight')

# %%
