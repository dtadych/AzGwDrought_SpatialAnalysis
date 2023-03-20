#%%
#import all necessary packages
#RUN THIS CODE IN 'research' ENVIRONMENT
import os
import numpy as np
#import csv
#import osmnx as ox
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.windows import crop, transform
import rioxarray as rxr
import rasterio as rio
import rasterstats
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.plot import plotting_extent
from rasterio.warp import calculate_default_transform, reproject, Resampling
#from rasterstats import zonal_stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#import earthpy.plot as ep
#import shapely
#from glob import glob
#from PIL import Image

#when i import other packages (GDAL, OGR from OSGEO) it seems to mess up the shape file
#stick with what isnt commented out right now

#%%
#Read the Shapefile (check the input filepath to make sure you are importing the shapefile you want)
irr_dist = gpd.read_file(r"C:\Users\mpereira\Documents\GradSchool\Research\Data\Shapefiles\Groundwater_Subbasin\GW_Subbasins_StudyArea.shp")

#check the crs of the vector
irr_dist.crs

#%%
#create a loop for all the years of the .tif files
for y in range(2010,2021):              
    #read in the filename (change the directory to where your .tif files are)
    filein="C:/Users/mpereira/Documents/GradSchool/Research/Data/Cropscape/state=04_year="+str(y)+"_crop=all.tif"

    print(filein)
    print(y)

    #read in raster defined in filein
    alfalfa_2020 = rio.open(filein)

    #project shapefile to match raster
    irr_dist_match = irr_dist.to_crs(alfalfa_2020.crs)

    #create plotting extent from DatasetReader Object
    alfalfa_2020_plot_extent = plotting_extent(alfalfa_2020)

    #see coordinates of plotting extent of alfalfa raster
    alfalfa_2020_plot_extent

    #see object type
    #type(alfalfa_2020_plot_extent)

    #see bounds attribute
    alfalfa_2020.bounds

    #see the bounds type
    #type(alfalfa_2020.bounds)

    #Assign raster values to a numpy nd array
    alfalfa_2020_array= alfalfa_2020.read(1)
    alfalfa_2020_array.astype(np.float32)

    #Create a variable with the transform information needed for later
    affine = alfalfa_2020.transform

    #%%
    #plotting section of rasters and vectors and overlays
    #COMMENTED OUT FOR FASTER RUN TIME PURPOSES

    #plot .tif ontop of the shapefile
    #fig, ax=plt.subplots(figsize=(5,5))
    #show(alfalfa_2020, ax=ax, title="raster ontop of vector")
    #irr_dist_match.plot(ax=ax, facecolor='none', edgecolor='yellow')
    #plt.show

    #plotting a histogram of all the crop types on the tif(raster image)
    #fig, ax=plt.subplots()
    #show_hist(alfalfa_2020, ax=ax, title='Histogram')
    #plt.show

    #plot .tif by itsself
    #show(alfalfa_2020, title='Alfalfa 2020')

    #plot shapefile by ittself
    #irr_dist.plot()

    #%%
    #loop for a tif with all crops on it
    #1,2,4,6,12,14,21,22,23,24,27,28,29,36,37,41,42,43,44,47,48,49,51,53,54,57,59,61,67,68,69,71,72,74,77,111,121,122,123,124,131,141,142,143,152,176,190,195,204,205,206,207,208,209,211,212,213,214,215,216,219,225,226,227,228,229,231,232,233,236,237,238,243,244,245,246,249  
    crop_list=[1,2,4,6,12,14,21,22,23,24,27,28,29,36,37,41,42,43,44,47,48,49,51,53,54,57,59,61,67,68,69,71,72,74,77,111,121,122,123,124,131,141,142,143,152,176,190,195,204,205,206,207,208,209,211,212,213,214,215,216,219,225,226,227,228,229,231,232,233,236,237,238,243,244,245,246,249]                                #create a list of all the crop types
  
    n_crop=len(crop_list)   #create a variable n_crop that contains the list crop_list
    print("crop_list length")
    print(len(crop_list))   #print the length of the crop list

    #ii=4   #used if you want to run it for 1 crop
    for ii in range (0,n_crop):                             #creating a loop from 0 to end of n_crop list
        crop_temp=crop_list[ii]                          #creating a temp variable of the crop type the loop is on 
        print("loop step","crop type")
        print(ii,crop_temp)                           #printing which i corresponds to which crop type 
        crop_tif=alfalfa_2020_array.copy()            #creating a copy of the array with all the crop types on it
        crop_tif[crop_tif!=crop_temp]=0                   #in the crop_tiff array all crop values that arent = to the current crop make 0
        crop_tif[crop_tif==crop_temp]=1                   #assign a value of 1 to the current crop the loop is on (this makes it so u dont have to divide by crop value)
        stats_temp=rasterstats.zonal_stats(irr_dist_match, crop_tif, affine=affine, stats=['count', 'sum'], geojson_out=True)  #creating a dataframe with the raster stats for that crop type
        #print(type(stats_temp))                         #if i want to find out what this output is
        zonal_stats=[]                                  #create a list called zonal stats with all 0s
        i=0

        while i < len(stats_temp):                              #take length of states_temp
            zonal_stats.append(stats_temp[i]['properties'])     #append the zonal stats list to stats_temp
            i = i+1

        zonal_stats_df = pd.DataFrame(zonal_stats)              #transferring the data to a pandas DF         
        #print(zonal_stats_df)                                   #print the zonal stats dataframe

        if ii==0:
            count=zonal_stats_df[['count']].copy()          #create a new DF called count w/ the count column from zonal_stats_df
            stats_summary=zonal_stats_df[['sum']].copy()    #create a new DF stats_summary where you take the sum from zonal_stats_df
            call_name="crop"+str(crop_temp)                 #create a column header for current crop type
            stats_summary.rename(columns={"sum":call_name},inplace=True) #replace "sum" column name to crop type name
        else:
            call_name="crop"+str(crop_temp)                           ##create a column header for current crop type
            stats_summary[call_name]=zonal_stats_df[['sum']]          #add the sum column to the stats_summary DF with the current crop type
                                                                         
    stats_summary.index=zonal_stats_df['SUBBASIN_1']                          #changing the index of stats_summary to county names
    stats_summary.to_csv('C:/Users/mpereira/Documents/GradSchool/Research/INFEWS_Codes/Cropscape_Processing_StudyArea/state=AZ_year={0}_crop=all.csv'.format(y))

count.index=zonal_stats_df['SUBBASIN_1']                                          #changing the index of count to county names
count.to_csv('C:/Users/mpereira/Documents/GradSchool/Research/INFEWS_Codes/Cropscape_Processing_StudyArea/count.csv')

#print(stats_summary)

#%%