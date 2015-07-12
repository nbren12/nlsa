from nlsa.diffusionmap import pdist_dask, compute_kernel, embed_pdist, compute_autotuning
import h5py
import xray
import numpy as np
import pickle
from scipy.sparse.linalg import eigsh
import imp

data = imp.load_source("data", "data.py")
configfile: "nlsa.yaml"


# Before Change directory
workdir: "anl"

rule all:
    input: "wthermo/q20/e1_a1/eigs.pkl"

rule eigs2orthog:
    input: "eigs.pkl"
    output: "orthog.pkl"

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
    output: "{tag}/data.npz"
    params: tag="{tag}"
    run:
        datakw = config['data'][params.tag]
        varname = datakw[ 'var' ]
        tdim = datakw['tdim']

        v = getattr(data, varname)
        v = v.fillna(0.0)

        nt = len(v.coords[tdim])
        out = np.reshape(v.values, (nt, -1))
        np.savez(output[0], out)
