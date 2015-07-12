from toolz import *
import numpy as np
from numpy.linalg import svd, lstsq


def project_lag(metric, phi, data, lag):
    """Project lags"""

    data.index += lag
    data, phi = data.align(phi, join='inner', axis=0)
    data.fillna(0.0, inplace=True)

    phi_w = phi.values * np.sqrt(metric)[:,None]
    data_w = data.values * np.sqrt(metric)[:,None]
    # phi_w = phi.values * metric[:,None]
    # data_w = data.values * metric[:,None]

    return data.index, lstsq(phi_w, data_w)[0]

def concatlags(A):
    return np.vstack(aa[1][None, ...] for aa in A)

def make2d(x):
    nlag, nphi = x.shape[:4]
    x = x.transpose((1, 0, 2, 3))
    x = np.reshape(x, (nphi, -1))
    x = x.T
    return x


@curry
def make2d_i(nlag, shape, x):
    
    return x.reshape((nlag,) + shape)

def fillna(x):
    x[np.isnan(x)] = 0.0
    return x

    

@curry
def selcol(cols, x):
    return x[...,cols]

@curry
def recon(phi, A, linmap):
    import dask.array as da
    nlag = len(A)

    X = pipe(A, concatlags, make2d, selcol(linmap), fillna)
    X.shape = (nlag, -1, X.shape[-1])
    Xd = da.from_array(X, chunks=(100,100, X.shape[-1]))
    phid = da.from_array(selcol(linmap, phi), chunks=(100,100, X.shape[-1]))

    recon = Xd.dot(phid.T)
    nlag, n, ns = recon.shape
    out = np.zeros((n, ns + nlag))


    for lag in range(nlag):
        print("Processing lag %d"%lag)
        inds = slice(nlag-lag-1, ns+nlag-lag-1)
        out[:, inds] += recon[lag, :, :]
    return out


@curry
def flat2d(A, x):
    return np.reshape(x, A[0][1].shape[1:])


def pdist(A):
    """
    Args:
        A (s, n): array-like to perform pairwise distance of

    Returns:
        c (s, s): dask-array with pdists
    """
    import dask.array as da

    # Load A as a dask array
    s, n = A.shape
    nchunk  = int(1e8 // A.dtype.itemsize  // max(s,n)) # 1 MB chunks approximately
    chunks = (nchunk, n)
    Ad = da.from_array(A, chunks=chunks)

    # Make dask graph for pdist
    b = Ad.dot(Ad.T)
    mag = (Ad ** 2).sum(axis=1)

    c = mag[:,None] + mag[None,:] - 2*b

    return c

def test_pdist():
    n = 10000
    s = 500
    A = np.random.rand(s, n)

    c = pdist(A)
    c.to_hdf5("out.h5", "/pdist")





svd =curry(svd)

if __name__ == '__main__':
    test_pdist()

