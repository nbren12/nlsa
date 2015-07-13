import os

import numpy as np

import h5py
import xray
import pandas as pd



# Load isotemp data 

root = os.path.dirname(__file__)
nc = "/home/noah/scratch/rayben_clean/data/2013-07-19/isotherm/bin15.nc"
isotemp = xray.open_dataset(nc)
isotemp['t'] = np.arange(len(isotemp['t'])) + 10000
isotemp.coords['z']= np.linspace(0, 1, len(isotemp.z))
temp = isotemp.coords['temp']
temp -= .5
temp += (temp[1] -temp[0]) /2
isotemp.coords['temp']  = temp
w = isotemp['w']
w -= w.mean('t')
