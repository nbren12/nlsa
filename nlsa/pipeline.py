import json, os

__author__ = 'noah'
import pickle
import re
import numpy as np, pandas as pd
import xray


from .data import phi, metric
from .nlsa import project_lag
import sys

def get_lag_from_fn(filename):

    lag = re.search(r"(\d*?)\.amat", filename).group(1)
    lag = int(lag)

    return lag

# The project the data
def mk_amat(output_filename, data, lag, phi, field='T'):
    """
    datafile is an xray object with 't' dimension first

    TODO: this step uses a ton of memory! use on disk arrays maybe.
    """
    # 
    
    # Convert to pandas for helpful alignment
    # raw = xray.open_dataset(datafile)[field]

    # Flatten data (assume 't' is first)
    arr = np.reshape(data.values, (data.shape[0], -1))
    data = pd.DataFrame(arr, index=data.coords[data.dims[0]])


    out = project_lag(phi.metric, phi, data, lag)
    with open(output_filename, "wb") as f:
        pickle.dump(out, f)


def recon_lag(idx, A, time_range=(0,100), linmap=(0,), phi=None, **kw):

    X = A[linmap, :]
    philoc = phi.ix[:, linmap].set_index(idx)
    mu0    = phi.metric
    mu0.index = idx

    # Subset
    idx = idx[ (idx >= time_range[0]) & (idx < time_range[1])]
    philoc = philoc.ix[idx].values /mu0[idx].values[:,None]

    return idx, philoc.dot(X)

def recon_all(inputs, orthog, output, **kw):
    import pandas as pd
    from functools import reduce


    phi = pd.read_pickle(orthog)

    def fn2df(fn):
        with open(fn, "rb") as f:
            idx, A = pickle.load(f)

        idx, data = recon_lag(idx, A, phi=phi, **kw)
        return pd.DataFrame(data, index=idx)

    def adddf(a,b):
        a, b = a.align(b, fill_value=0)
        return a+b

    dfs = (fn2df(fn) for fn in inputs)

    out = reduce(adddf, dfs)/ len(inputs)

    out.to_pickle(output)


def df2xray(t, arr, base: xray.DataArray, name='w') -> xray.DataArray:

    if t is None:
        nt = arr.shape[0]
        t  = np.arange(nt)
    else:
        nt = len(t)

    shape = (nt,) + base.shape
    arr = np.reshape(arr, shape)

    dims = ('t',) + base.dims
    coords = {key: np.asarray(base.coords[key]) for key in dims}
    coords['t'] = np.asarray(t)

    return xray.DataArray(arr, coords=coords, name=name, dims=dims)
