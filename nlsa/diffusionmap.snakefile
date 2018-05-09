from nlsa.io import get_data
from nlsa.diffusionmap import (pdist_dask, compute_kernel,
                               embed_pdist_fast as embed_pdist,
                               compute_autotuning,
                               symmetric2orthog, symmetric)
import h5py
import xarray as xr
import numpy as np
import pickle
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix
import imp



rule eigs2orthog:
    input: eigs="{tag}/{a}/E{b,[^/]*}/eigs.pkl", time="{tag}/time.npz"
    output: "{tag}/{a}/E{b,[^/]*}/orthog.pkl"
    run:
        import pandas as pd

        d = np.load(input.eigs)
        t = np.load(input.time)['arr_0']

        if config['diffmaps'][wildcards.b]['symmetric_eigs']:
            phi = symmetric(d['phi'], t)
        else:
            phi = symmetric2orthog(d['phi'], t)

        phi.dropna().to_pickle(output[0])



rule eigs:
    input: "{dir}/E{a}/K.npz"
    output: "{dir}/E{a}/eigs.pkl"
    run:
        # num_neighbors = 5000


        # process configurations
        try:
            neig = config['diffmaps'][wildcards.a]['n']
        except:
            neig = 100

        try:
            num_neighbors = config['diffmaps'][wildcards.a]['num_nearest']
        except:
            num_neighbors = None # if num neigbors

        try:
            sparsity = config['diffmaps'][wildcards.a]['sparsity']
        except:
            sparsity = None # if num neigbors

        K = np.load(input[0])['arr_0']

        # Find nan mask for phi output
        ## Get first row of K which isn't all NAs
        ind = np.any(~np.isnan(K), axis=-1).nonzero()[0][0]
        mask = np.isnan(K[ind])

        # Fill na with zero
        K[np.isnan(K)] = 0

        # Nearest neighbors
        if num_neighbors:
            inds = K.argsort(axis=1)
            num_no_neighbs = K.shape[0] - num_neighbors
            xind = np.arange(inds.shape[0])[:,None]
            yind  = inds[:,:num_no_neighbs]
            K[xind, yind] = 0

            K  = csc_matrix(K)

            # Symmetrize
            K = ( K + K.T)/2
        elif sparsity is not None:
            K[K < np.percentile(K, 100*(1-sparsity))] = 0.0
            K  = csc_matrix(K)

        # Eigenvalue problem
        lam, phi = eigsh(K, k=neig)

        # Fill in NA
        phi[mask,:] = np.nan
        pickle.dump(dict(lam=lam[::-1], phi=phi[:,::-1]),
                    open(output[0], "wb"))


rule kernel:
    input: at="{dir}/at.npz", dist="{dir}/emb_pdist.npz"
    output: "{dir}/E{a}/K.npz"
    run:
        diffmap = config['diffmaps'][wildcards.a]
        alpha = diffmap['alpha']
        eps = diffmap['eps']

        xi = np.load(input.at)['arr_0']
        dist = np.load(input.dist)['arr_0']
        K = compute_kernel(dist, xi, eps)


        norm = np.nansum(K, axis=0)
        norm = norm**alpha
        norm[ norm < 1e-9] = 1.0 # Fill in ones to avoid /0 error

        K /= norm[:,None]*norm[None,:]

        np.savez(output[0], K)

rule embed_dist:
    input: "{tag}/pdist.npz"
    output: "{tag}/q{q}/emb_pdist.npz"
    params: q="{q}"
    run:
        C = np.load(input[0])['arr_0']
        Cemb = embed_pdist(C, int(params.q))
        np.savez(output[0], Cemb)
rule at:
    input: "{dir}/emb_pdist.npz"
    output: "{dir}/at.npz"
    run:
        C = np.load(input[0])['arr_0']
        at = compute_autotuning(C)
        np.savez(output[0], at)


rule pdist:
    input: "{tag}/data.npz"
    output: "{tag}/pdist.npz"
    # shell: "python -m nlsa.pdist {input} {output}"
    run:
        X = np.load(input[0])['arr_0']
        P = X.dot(X.T)
        D = np.diag(P)

        C = D[:,None] + D[None,:] -2*P
        C[C <0] =0.0
        np.savez(output[0], C)


rule data:
    output: data="{tag}/data.npz", time="{tag}/time.npz"
    params: tag="{tag}"
    run:
        v, tdim = get_data(config, wildcards.tag)
        v = v.fillna(0.0)

        nt = len(v.coords[tdim])

        out = np.reshape(v.values, (nt, -1))
        print(output.data)
        np.savez(output.data, out)


        time = np.asarray(v.coords[tdim])
        np.savez(output.time, time)
