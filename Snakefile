from src import pdist_dask, compute_kernel, embed_pdist, compute_autotuning
import h5py
import xray
import numpy as np
import pickle
from scipy.sparse.linalg import eigsh


rule eigs:
    input: "K.npz"
    params: neig="100"
    output: "eigs.pkl"
    run:
        neig = int(params.neig)
        K = np.load(input[0])['arr_0']

        # Find nan mask for phi output
        ## Get first row of K which isn't all NAs
        ind = np.any(-np.isnan(K), axis=-1).nonzero()[0][0]
        mask = np.isnan(K[ind])

        # Fill nas with 0
        K[np.isnan(K)] = 0.0

        # Eigenvalue problem
        lam, phi = eigsh(K, k=neig)

        # Fill in NA
        phi[mask,:] = np.nan
        pickle.dump(dict(lam=lam[::-1], phi=phi[:,::-1]),
                    open(output[0], "wb"))


rule kernel:
    input: at="at.npz", dist="emb_pdist.npz"
    output: "K.npz"
    params: eps="1.0", alpha="1.0"
    run:
        alpha = float(params.alpha)

        xi = np.load(input.at[0])['arr_0']
        dist = np.load(input.dist[0])['arr_0']
        K = compute_kernel(dist, xi, float(params.eps))


        norm = K.sum(axis=0)
        norm = norm**alpha

        K / norm[:,None]/ norm[None,:]
        np.savez(output[0], K)

rule embed_dist:
    input: "pdist.npz"
    params: q="200"
    output: "emb_pdist.npz"
    run:
        C = np.load(input[0])['arr_0']
        Cemb = embed_pdist(C, int(params.q))
        np.savez(output[0], Cemb)
rule at:
    input: "emb_pdist.npz"
    output: "at.npz"
    run:
        C = np.load(input[0])['arr_0']
        at = compute_autotuning(C)
        np.savez(output[0], at)


rule pdist:
    input: "data.npz"
    output: "pdist.npz"
    shell: "python pdist.py {input} {output}"


rule data:
    input: "bin15.nc"
    params: var='w', tdim ='t'
    output: "data.npz"
    run:
        x = xray.open_dataset(input[0])

        v = x[params.var]
        v = v.fillna(0.0)
        v -= v.mean(params.tdim)

        nt = len(v.coords[params.tdim])
        out = np.reshape(v.values, (nt, -1))
        np.savez(output[0], out)
