# %%
# import all packages needed to run code
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob

# %%
# Read in pump_data_Full file
# This file has well id as rows along with 24 columns of other variables (Af Pumped=most important)
filepath = os.path.join('/Users/matthewford/Desktop/Python files/Output_files/Pump_Data_Full.csv')
print(filepath)

#%%
#make a datatable called cadastral_data with the read in excel file
pump_data_all = pd.read_csv(filepath)

#make scientific notation go away (needed for wellid column)
pd.options.display.float_format = '{:.2f}'.format
print(pump_data_all.info())

#%%
# Create a pivot table of AF Pumped by well id and year
# This combines all common well ids and averages all observations by year
pivot1 = pd.pivot_table(pump_data_all, index=['wellid','YEAR'], values='AF Pumped')
print(pivot1)

#%% 
# Save pivot table 1 to a csv in the specified directory 
pivot1.to_csv('/Users/matthewford/Desktop/Python files/Output_files/AFPumped_Bywellid&Year.csv')

#%%
# create a dataframe  which unstacks pivot table 1 so wellid=row & columns are years 
pump_data1 = pivot1.unstack(level=1)
print(pump_data1)

#%% 
# Save pump_data1 to a csv in the specified directory 
pump_data1.to_csv('/Users/matthewford/Desktop/Python files/Output_files/AF_Pump_byyear_unstack.csv', float_format='%g')

#%%
# Find the number of non NAN values 
# This shows total number of obversations(by year) we have for each well id
nonnanvalues = pump_data1.isnull().sum(axis=1)
print(nonnanvalues)

#%%
#Create a variable called "myid" to make locating graphing wells easier
myid=313658111031201

# %%
# take pivot1 and locate all the AF Pumped by year for the given wellid above
pivot1.loc[myid]

#%%
# plot the listed well id by wl depth and year and maniplulate the graph
fig, ax = plt.subplots()
ax.plot(pivot1.loc[myid],label=myid)
ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
plt.ylim()
plt.xlim(1984,2020)
ax.grid()
ax.legend()
plt.show


#Save plot as a png with the myid name to the specified directory
type = myid
plt.savefig('/Users/matthewford/Desktop/Python files/Output_files/{0}.png'.format(type), bbox_inches='tight')

#%%
# Create a pivot table summing up the AF Pumped by year by basin
pivot2 = pd.pivot_table(pump_data_all, index=['Basin','YEAR'], values='AF Pumped',aggfunc=np.sum)
print(pivot2)

#%%
#Unstack pivot 2 into a new dataframe pump_data2 with basin as rows and years as columns
pump_data2 = pivot2.unstack(level=1)
print(pump_data2)

#%%
# create and locate a variable for the basin we want to look at
mybasinid='TUCSON AMA'
pivot2.loc[mybasinid]

# %%
# plot the basin id by AF Pumped & year 
fig, ax = plt.subplots()
ax.plot(pivot2.loc[mybasinid], label=mybasinid)
ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
ax.legend()
ax.grid()
plt.show

#Save plot as a png with the myid name to the specified directory
type = mybasinid
plt.savefig('/Users/matthewford/Desktop/Python files/Output_files/{0}.png'.format(type), bbox_inches='tight')

#%%
#Unstack pivot 2 into new df called pump_data1 (now row=year and column is the basinid)
pump_data3 = pivot2.unstack(level=0)
print(pump_data3)

#%%
#Create a list of basin ids so graph can make a legend
mylist=pump_data_all['Basin'].unique()
print(mylist)

#%%
#Create a loop where AF Pumped is plotted by basin
fig, ax = plt.subplots()
for i in range(len(mylist)):
    print(i)
    print(mylist[i])
    ax.plot (pump_data3['AF Pumped'][mylist[i]], label=mylist[i])


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
ax.grid()
plt.show

#Save plot as png to the specified directory
plt.savefig('/Users/matthewford/Desktop/Python files/Output_files/AFPumped_ByBasin.png', bbox_inches='tight')

#%%
#Create a loop of multiple plots of AF pumped for each basin 
fix, axes = plt.subplots(figsize=(20,20),nrows=4, ncols=4, sharex=True, sharey=True)
axes_list = [item for sublist in axes for item in sublist]

for i in range(len(mylist)):
    ax=axes_list.pop(0)
    ax.plot(pump_data3['AF Pumped'][mylist[i]])
    ax.set_title(mylist[i])
    ax.set_xlim((1980,2020))
    ax.set_xticks(range(1980, 2020, 10))
    ax.set_ylim((0,20000))
    ax.grid(linewidth=0.25)
    ax.set_xlabel('Year')
    ax.set_ylabel('AF Pumped')
    ax.tick_params(
        which='both',
        bottom='off',
        left='off',
        right='off',
        top='off',
    )

for ax in axes_list:
    ax.remove()

plt.tight_layout()

#Save plot as png to the specified directory
plt.savefig('/Users/matthewford/Desktop/Python files/Output_files/AFPump_individualbasins.png', bbox_inches='tight')

# %%
