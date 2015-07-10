from src import pdist_dask
import xray
import numpy as np
import sys

input  = sys.argv[1]
output = sys.argv[2]


X = np.load(input)['arr_0']

C = pdist_dask(X)
C.to_hdf5(output, "/dist")

