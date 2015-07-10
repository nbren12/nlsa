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
    Cout = np.empty_like(C)
    Cout[:] = np.nan

    for i in range(q, s):
        for j in range(q, s):
            Cout[i,j] = 0.0
            for k in range(q):
                Cout[i,j] += C[i-k, j-k]




    return Cout

@autojit
def compute_autotuning(C):
    """
    Returns:
    xi:  |x_i- x_{i-1}|^2
    """
    s = C.shape[0]
    at = np.empty_like(C[0])
    at[:] = np.nan

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

