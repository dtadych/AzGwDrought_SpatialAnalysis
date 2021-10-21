# AzGwDrought_SpatialAnalysis

This repository contains folders for our Spatial Analysis of Wells in Arizona.

Folders:
- FromMatt -> Contains files from Matt's compiling of the GWSI database
  1. Manipulation_files copy: Manipulates GWSI databases and creates graphs
      - Pump_data_manipulate.py: Creates GWSI pumping databases and graphs
- MergedData -> Contains files from merged GWSI database, shapefiles from Wells55 and GWSI
  1. Output_files: any outputs from the summary SummaryCodes folder
  2. Shapefiles: input shapefiles for the geoprocessing analysis in SummaryCodes
- SummaryCodes -> Contains python scripts to work with Wells55 database, GWSI, Summary and manipulation codes but edited by Danielle, and code to combine databases
  1. Data_Tables: contains GWSI excel sheets
  2. Scripts:
      - Wells55_GWSI_Static_Merge: This script combines the GWSI database and the Wells55 database.
          - In order for this script to work, in the
      - Georegion_merge.py: This script is used to combine separate shapefiles into one shapefile in order to overlay on top of the master database.  There is a step in QGIS in order to complete the merged
      - Spatial_Analysis.py: this code is used to overlay the georegion shapefile, read in the timeseries database, and create averaged timeseries water level databases based on the region
          - In order for this to work, in the Output_files folder there needs to be
            1. Master_ADWR_Database_v# (created from Wells55_GWSI_Static_Merge)
            2. Georegions_fixed (whatever shapefile created for overlay created from Georegion_merge)
            3. Wells55_GWSI_WLTS_DB_annual.csv (Annual Timeseries water level database created from Well_Timeseries_merge.py)

      - pump_data_manipulate_DT.py: the same as Matt's but with my own edits
      - pump_wl_combined_manipulate_working: the same as Matt's but with my own edits
