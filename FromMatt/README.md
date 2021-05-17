# Read_Me

These python files can be found on my [AZGW_research](https://github.com/mford4/AZGW_research)

### Merge Files:
- `Merge_WL_w_basinid.py`
	Imports `GWSI_WW_LEVELS.xlsx` containing depth to water and merges it on "basinid" with `GWSI_SITES.xlsx` and exports  `wl_data2.csv` to a csv in the specified directory.
	1. Download `GWSI_WW_LEVELS.xlsx` & `GWSI_SITES.xlsx` at [Google Drive](https://drive.google.com/drive/u/0/folders/1J3SjjTUHMRXKCPFWZa0yb-22okZGJ1N4)
	2. Change the input directories (lines 15 & 29) to the location of those xlsx files on your personal computer.
	3. Change the output directory to the location on your computer where you want the files saved to (line 48)
	4. Run code

- `pump_data_concat.py`
	Imports all 7 pump data csv files and merges them on "cadastralid" so they are now linked with wellid and output a csv `pump_data_full.csv`.
	1. Download `pump_data_1984-1989.csv`,`pump_data_1990-1995.csv`,`pump_data_1996-2000.csv`, `pump_data_2001-2005.csv`, `pump_data_2006-2010.csv`, `pump_data_2011-2015.csv`, `pump_data_2016-2020.csv` & `GWSI_SITE_CADASTRAL_HISTORY.xlsx` at [Google Drive](https://drive.google.com/drive/u/0/folders/1J3SjjTUHMRXKCPFWZa0yb-22okZGJ1N4)
	2. Change the input directories (lines 14 & 27) to the location of those csv & xlsx files on your personal computer.
	3. Change the output directory to the location on your computer where you want the files saved to (line 49)
	4. Run code

- `pumpdata_join_welldata.py`
	Imports `Pump_Data_Full.csv` and imports the `wl_data2.csv` and merges them on "wellid" and creates `pump_wl.csv`.
	1. Download `Pump_Data_Full.csv` & `wl_data2.csv` at [Google Drive](https://drive.google.com/drive/u/0/folders/1J3SjjTUHMRXKCPFWZa0yb-22okZGJ1N4)
	2. Change the input directories (lines 12 & 22) to the location of those csv files on your personal computer.
	3. Change the output directory to the location on your computer where you want the files saved to (line 37)
	4. Run code

### Manipulation Files:
- `pump_data_manipulate.py`
	Imports `pump_data_full.csv` and manipulates it to make graphs related to the amount of Acre Feet of water pumped in wells in AZ. Data obtained from ADWR.
	1. Download `Pump_Data_Full.csv` at [Google Drive](https://drive.google.com/drive/u/0/folders/1J3SjjTUHMRXKCPFWZa0yb-22okZGJ1N4)
	2. Change the input directory (line 12) to the location of that csv files on your personal computer.
	3. Change all output directories
	4. Run code

- `pump_wl_combined_manipulate.py`
	Imports `Pump_wl.csv` file and manipulates it to make graphs
	1. Download `Pump_wl.csv` at [Google Drive](https://drive.google.com/drive/u/0/folders/1J3SjjTUHMRXKCPFWZa0yb-22okZGJ1N4)
	2. Change the input directory (line 12) to the location of that csv files on your personal computer.
	3. Run code


- `WL_Data_Eval.py`
	Imports the `wl_data2.csv` file and manipulates it to make graphs related to the depth to water in wells in AZ. Data obtained from ADWR.
	1. Download `wl_data2.csv` at [Google Drive](https://drive.google.com/drive/u/0/folders/1J3SjjTUHMRXKCPFWZa0yb-22okZGJ1N4)
	2. Change the input directory (line 12) to the location of that csv files on your personal computer.
	3. Change all output directories
	4. Run code
