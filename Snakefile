from src import pdist_dask, compute_kernel, embed_pdist, compute_autotuning
import h5py
import xray
import numpy as np


rule kernel:
    input: at="at.npz", dist="emb_pdist.npz"
    output: "K.npz"
    params: eps="1.0"
    run:
        xi = np.load(input.at[0])['arr_0']
        dist = np.load(input.dist[0])['arr_0']
        K = compute_kernel(dist, xi, float(params.eps))
        np.savez(output[0], K)

rule embed_dist:
    input: "pdist.h5"
    params: q="20"
    output: "emb_pdist.npz"
    run:
        C = h5py.File(input[0])['/dist'][:]
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
    output: temp("pdist.h5")
    shell: "python pdist.py {input} {output}"


rule data:
    input: "bin15.nc"
    params: var='w', tdim ='t'
    output: "data.npz"
    run:
        x = xray.open_dataset(input[0])

        nt = len(x.coords[params.tdim])
        out = np.reshape(x[params.var].values, (nt, -1))
        np.savez(output[0], out)
