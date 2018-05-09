"""
Pipeline for reconstruction of the Raleigh Benard data.

This code can easily be generalized for arbitrary datasets.
As input it uses a json file, which specifies some basic details.

Input is specified using:
    "obs": [modulename, dict, varname]
where modulename.dict[varname] points to an xray dataarray with a temporal dimension 't'.

"""
from nlsa.pipeline import *
from nlsa.io import get_data
import pandas as pd
import os, pickle, re
import importlib

#from gnl.xarray import *


#------------------------------------------------------------
#              Reconstruction rules
#------------------------------------------------------------


def nc2dfpair(nc):

    wT = xarray.open_dataset(nc)
    wT['half'] = wT.z >= .5

    arr = wT.groupby('half')\
            .apply(lambda x: integrate(x['wT'], axis='z'))
    df = pd.DataFrame(arr.values.T, index=arr.coords['time'])
    
    return df

# Generate two time series for heat flux recon.
#
# 1. Heat flux with z >= .5
# 2. Heat flux for  z < .5
rule wT2pair:
    input: "wthermo/{linmap}/wT.nc"
    output: "wthermo/{linmap}/flux_pair.pkl"
    run:
        nc2dfpair(input[0]).to_pickle(output[0])



rule wT:
    input: "wthermo/{linmap}/recon.nc"
    output: "wthermo/{linmap}/wT.nc"
    run:
        wT(xarray.open_dataset(input[0])['w'])\
            .rename('wT')\
            .to_dataset()\
            .to_netcdf(output[0])

rule svd:
    input: "{dir}/{tag}/amat.nc", phi="{dir}/orthog.pkl"
    output: a="{dir}/{tag}/S{svdspec,[^/]*}/svd.pkl",\
            o="{dir}/{tag}/S{svdspec,[^/]*}/orthog.pkl"
    run:
        from numpy.linalg import svd
        amat = xarray.open_dataset(input[0])['amat']

        spec = config['svds'][wildcards.svdspec]
        inds = range(spec)

        # Load eigenvalues
        phi =  pd.read_pickle(input.phi)
        amat = amat.sel(eignum=inds)

        # Turn amat into flat array
        dims = [dim for dim in amat.dims]

        ## Move eignum dim to end
        dims.remove('eignum')
        dims.append('eignum')
        amat = amat.transpose(*dims)

        ## Reshape data
        neig = amat.shape[-1]
        amat = np.reshape( amat.values, (-1, neig))

        U, S, V = svd(amat, full_matrices=False)
        vT  = phi.ix[:,inds].dot(V.T)

        pickle.dump( (U, S, V, vT), open(output.a, "wb"))

        vT['metric'] = phi.metric
        vT.dropna().to_pickle(output.o)

def alags_inputs(wildcards):
    # Search for number of lags in names of parent dirs
    d = wildcards.dir
    pat = re.compile("q(\d+)")
    lags = None
    for subdir in d.split('/'):
        match = pat.match(subdir)
        if match:
            lags =  int(match.group(1))
            break

    # if no lags found the raise error
    if not lags:
        raise ValueError

    # if lags make file
    return [ os.path.join(wildcards.dir, wildcards.tag, "%d.amat.pkl"%lag)
            for lag in range(lags)]
        


rule alags:
    # input: expand("{{dir}}/{{tag}}/{lag}.amat.pkl", lag=range(config['lags']))
    input: alags_inputs
    output: "{dir}/{tag}/amat.nc"
    run:
        tag= wildcards.tag
        base = get_data(config, tag)[0].isel(time=0)
        As = []
        lags = []

        for i in input:
            lag  = re.search("(\d+)\.amat\.pkl", i).group(1)
            lag  = int(lag)
            ind, A = pickle.load(open(i, "rb"))
            xd = df2xarray(None, A, base, name='amat')\
                 .assign_coords(lag=lag)\
                 .rename({'time':'eignum'})

            As.append(xd)

        xarray.concat(As, 'lag').to_dataset().to_netcdf(output[0])
        


rule pkl2nc:
    input: "{dir}/{tag}/R{recon}/recon.pkl"
    output: "{dir}/{tag}/R{recon}/recon.nc"
    run:
        tag= wildcards.tag
        base = get_data(config, tag)[0].isel(time=0)
        df = pd.read_pickle(input[0])
        df2xarray(df.index, df.values, base, name=tag)\
            .to_dataset()\
            .to_netcdf(output[0])


def get_recon(tag, rdir, config):
    for k in config['recons']:
        if rdir == k.format(tag=tag):
            print(k.format(tag=tag))
            return config['recons'][k]

rule recon_all:
    input: a=alags_inputs,\
           o="{dir}/orthog.pkl"
    output: pkl="{dir}/{tag}/R{recon}/recon.pkl"
    run:
        pkl = output.pkl
        rc = get_recon(wildcards.tag, os.path.dirname(pkl), config)
        recon_all(input.a, input.o, pkl, **rc)



rule split:
    input: "{orthog}/orthog.pkl"
    output: "{orthog}/{tag}/{lag}.amat.pkl"
    run:
        try:
            os.mkdir(wildcards.tag)
        except:
            pass
        tag = wildcards.tag
        base, t = get_data(config, tag)
        phi  = pd.read_pickle(input[0])
        mk_amat(output[0], base, int(wildcards.lag), phi, field='w')

