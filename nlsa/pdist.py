from .diffusionmap import pdist_dask
import xarray
import numpy as np
import sys

input  = sys.argv[1]
output = sys.argv[2]


X = np.load(input)['arr_0']

C = pdist_dask(X)
C = np.asarray(C)
np.savez(output, C)
