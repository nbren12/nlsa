import numpy as np
import dask.array as da
from numba import autojit
from math import exp, sqrt

def pdist_dask(X, cov=False):
    """Return dask array with ||x_i-y_i||^2 """
    s, n = X.shape
    Xd = da.from_array(X.astype(np.float32), chunks=(100,n))
    P = Xd.dot(Xd.T)
    D = (Xd*Xd).sum(axis=1)

    if cov:
        return P
    else:
        return D[:,None] + D[None,:] - 2*P

@autojit
def embed_pdist(C, q):
    """
    Given matrix of sum of squares compute sum accross lagged embedding
    """

    s = C.shape[0]
    Cout = np.ones_like(C) *np.nan
    Cout[:] = np.nan

    for i in range(q, s):
        for j in range(q, s):
            Cout[i,j] = 0.0
            for k in range(q):
                Cout[i,j] += C[i-k, j-k]




    return Cout

@autojit
def embed_pdist_fast(C,q):
    """
    A faster implementation of embed_pdist which may suffer from have
    cancellation floating point errors.

    This algorithm is O(s^2 + s^2). The other algorithm is O(q * s^2).

    Works by commulatively summing along the diagonals and then differencing
    the sum. This code yields identical results for 10000x10000 matrices, but
    if numerical cancellations become an issue, a block version of this
    algorithm may be necessary.
    """
    s,s1 = C.shape
    Cout = np.ones_like(C) *np.nan

    # sum along diagonals storing data in input array
    for i in range(1,s):
        for j in range(1,s1):
            C[i,j] += C[i-1,j-1]
        
        
    # calculate difference of summed data
    for i in range(q, s):
        for j in range(q, s1):
            Cout[i,j] = C[i,j] - C[i-q, j-q]
    return Cout


def test_embed_pdist():
    C = np.random.rand(1000,1000)
    q = 100

    D = embed_pdist(C, q)
    D1 = embed_pdist_fast(C.copy(), q)
    np.testing.assert_allclose(D, D1)
    


@autojit
def compute_autotuning(C):
    """
    uses second order centered difference

    Returns:
    xi:  |x_i- x_{i-1}|^2
    """
    s = C.shape[0]
    at = np.ones_like(C[0]) *np.nan

    for i in range(1,s-1):
        at[i] = C[i+1,i-1]

    return at


@autojit
def compute_kernel(C, xi, epsilon):
    s = C.shape[0]

    K = np.empty_like(C)

    for i in range(s):
        for j in range(s):
            K[i,j] = exp(-C[i,j] /  sqrt(xi[i]) / sqrt(xi[j]) / epsilon)

    return K



from scipy import sparse
def sparsify(K, nearest=4000):
    inds   = K.argsort()
    thresh =  K[np.arange(K.shape[0]), inds[:,nearest]]
    mask   = K < thresh[:,None]

    return mask


def symmetric2orthog(phi, t):
    """convert symmetric eigenfunctions to laplacian eigenfunctions"""
    import pandas as pd
    metric = phi[:,0].copy()**2
    tot = metric[-np.isnan(metric)].sum()
    metric /= tot
    phi = phi[:,1:]/phi[:,0][:,None]

    phi=  pd.DataFrame(phi, index=t)
    phi['metric'] = metric

    return phi

def symmetric(phi, t):
    import pandas as pd

    phi=  pd.DataFrame(phi, index=t)
    phi['metric'] = 1.0
    phi.metric /= phi.metric.sum()
    return phi
