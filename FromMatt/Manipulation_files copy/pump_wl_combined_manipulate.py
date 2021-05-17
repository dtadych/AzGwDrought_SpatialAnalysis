# %%
# import all packages needed to run code
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

#%%
# Read in pump_wl 
# This is a combined file with pump data & depth to water
filepath = os.path.join('/Users/matthewford/Desktop/Python files/Output_files/pump_wl.csv')
print(filepath)

pump_wl = pd.read_csv(filepath)
pd.options.display.float_format = '{:.2f}'.format
print(pump_wl.info())

# %%
