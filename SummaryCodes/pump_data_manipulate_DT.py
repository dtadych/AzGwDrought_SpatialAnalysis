# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob

# %%
# Read in pump_data_Full file
filename = 'Pump_Data_Full.csv'
datapath = '../MergedData'
outputpath = '../MergedData/Output_files'
filepath = os.path.join(datapath, filename)
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
pivot1.to_csv(outputpath + 'AFPumped_Bywellid&Year.csv')

#%%
# create a dataframe  which unstacks pivot table 1 so wellid=row &
pump_data1 = pivot1.unstack(level=1)
print(pump_data1)

# %%
# DT - Made a datable counting non-nan values
# - this way includes naan values, keeping it here so I learn -
# countpump = pd.pivot_table(pump_data_all, index=['wellid'], values='AF Pumped', aggfunc=len)
# print(countpump)
# - -

countpump = pump_data1.count(axis=1)
countpump_2 = pd.DataFrame(countpump)
print(countpump_2)
# %%
countpump_2.columns = ['Measurements']
print(countpump_2)
#%%

countpump_2.to_csv(outputpath + 'AFPumped_Bywellid&Year_count.csv')


#%%
# Find the number of non NAN values 
nonnanvalues = pump_data1.isnull().sum(axis=1)
print(nonnanvalues)

#%%
#Create a variable called "myid" to make locating graphing wells easier
myid=345434110171201

# %%
# take pivot1 and locate all the AF Pumped by year for the given wellid
spec_well = pivot1.loc[myid]
print(spec_well.count())
count = int(spec_well.count())
print(count)

#%%
# plot the listed well id by wl depth and year and maniplulate the graph
fig, ax = plt.subplots()
ax.plot(spec_well, label=myid)
ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
plt.ylim()
plt.xlim(1984,2020)
ax.grid()
ax.legend()
plt.show

# %%
#Save plot as a png with the myid name to the specified directory
type = myid
plt.savefig(outputpath + '{0}.png'.format(type), bbox_inches='tight')

#%%
# Create a pivot table summing up the AF Pumped by year by basin
pivot2 = pd.pivot_table(pump_data_all, index=['Basin','YEAR'], values='AF Pumped',aggfunc=np.sum)
print(pivot2)

#%%
#Unstack pivot 2 into a new dataframe pump_data1 with basin and year
pump_data1 = pivot2.unstack(level=1)
print(pump_data1)

#%%
# create and locate a variable for the basin we want 
mybasinid='PHOENIX AMA'
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
plt.savefig(outputpath + '{0}.png'.format(type), bbox_inches='tight')

#%%
#Unstack pivot 2 into new df called pump_data1 
pump_data1 = pivot2.unstack(level=0)
print(pump_data1)

#%%
#Create a list of basin ids so graph can make a legend
mylist=pump_data_all['Basin'].unique()
print(mylist)

# %%
# Trying to get a count grouped by basin
#summary = pump_data_all
test = pump_data_all.groupby('Basin')['AF Pumped'].count()
test.head(1)
test

# %%
# Trying to create a summary plot by basin
fig, ax = plt.subplots()
ax.bar(pump_data_all['Basin'], pump_data_all['AF Pumped'].count(), align='center', alpha=0.5)
plt.show()

# %%
summary1 = pump_data1

# fig, ax = plt.subplots()
for i in range(len(mylist)):
    print(mylist[i])
    print(pump_data1['AF Pumped'][mylist[i]].count())
    count = pump_data1['AF Pumped'][mylist[i]].count()
    ax.bar(pump_data_all['Basin'], count)


# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
# ax.grid()
# plt.show()

# %%
# Trying a different way for pump stuff
summ2 = pump_data1
for i in range (len(mylist)):
    count = count + 1
    summ2['count'] = count
    print(summ2.head(10))

#%%
# Create a loop where AF Pumped is plotted by basin
fig, ax = plt.subplots()
for i in range(len(mylist)):
    print(i)
    print(mylist[i])
    ax.plot (pump_data1['AF Pumped'][mylist[i]], label=mylist[i])


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set(title='Acre Feet Pumped', xlabel='Year', ylabel='Volume Pumped (AF)')
ax.grid()
plt.show

#%%
#Save plot as png to the specified directory
plt.savefig(outputpath + 'AFPumped_ByBasin.png', bbox_inches='tight')

#%%
#Create a loop of multiple plots of AF pumped for each basin 
fix, axes = plt.subplots(figsize=(20,20),nrows=4, ncols=4, sharex=True, sharey=True)
axes_list = [item for sublist in axes for item in sublist]

for i in range(len(mylist)):
    ax=axes_list.pop(0)
    ax.plot(pump_data1['AF Pumped'][mylist[i]])
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
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for ax in axes_list:
    ax.remove()

plt.tight_layout()

#Save plot as png to the specified directory
# plt.savefig(outputpath + 'AFPump_individualbasins.png', bbox_inches='tight')

#idk how to get black square around the last graph
# %%
