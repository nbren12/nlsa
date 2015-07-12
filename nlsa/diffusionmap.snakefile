from nlsa.io import get_data
from nlsa.diffusionmap import pdist_dask, compute_kernel, embed_pdist, compute_autotuning
import h5py
import xray
import numpy as np
import pickle
from scipy.sparse.linalg import eigsh
import imp

rule eigs2orthog:
    input: eigs="{tag}/{a}/{b}/eigs.pkl", time="{tag}/time.npz"
    output: "{tag}/{a}/{b}/orthog.pkl"
    run: 
        import pandas as pd

        d = np.load(input.eigs[0])
        t = np.load(input.time[0])['arr_0']

        phi = d['phi']
        metric = phi[:,0].copy()**2
        tot = metric[-np.isnan(metric)].sum()
        metric /= tot
        phi = phi[:,1:]/phi[:,0][:,None]

        phi=  pd.DataFrame(phi, index=t)
        phi['metric'] = metric

        phi.dropna().to_pickle(output[0])



rule eigs:
    input: "{dir}/K.npz"
    params: neig="100"
    output: "{dir}/eigs.pkl"
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
    input: at="{dir}/at.npz", dist="{dir}/emb_pdist.npz"
    output: "{dir}/e{eps}_a{alpha}/K.npz"
    run:
        alpha = float(wildcards.alpha)
        eps = float(wildcards.eps)

        xi = np.load(input.at[0])['arr_0']
        dist = np.load(input.dist[0])['arr_0']
        K = compute_kernel(dist, xi, float(eps))


        norm = K.sum(axis=0)
        norm = norm**alpha

        K / norm[:,None]/ norm[None,:]
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
    shell: "python -m nlsa.pdist {input} {output}"


rule data:
    output: data="{tag}/data.npz", time="{tag}/time.npz"
    params: tag="{tag}"
    run:
        v, tdim = get_data(config, wildcards.tag)
        v = v.fillna(0.0)

        nt = len(v.coords[tdim])
        
        out = np.reshape(v.values, (nt, -1))
        np.savez(output.data[0], out)


        time = np.asarray(v.coords[tdim])
        np.savez(output.time[0], time)
