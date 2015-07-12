"""
Pipeline for reconstruction of the Raleigh Benard data.

This code can easily be generalized for arbitrary datasets.
As input it uses a json file, which specifies some basic details.

Input is specified using:
    "obs": [modulename, dict, varname]
where modulename.dict[varname] points to an xray dataarray with a temporal dimension 't'.

"""
from rayben.pipeline import *
from rayben.plots import *
from rayben.data import isotemp, phi, metric
import matplotlib as mpl
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle, re
import importlib

from gnl.xray import *

mpl.rc('savefig', bbox='tight', dpi=125)
mpl.rc('font', size=8, family='serif')
mpl.rc('axes.formatter', limits=(-1, 4))

figures= []

# Report
from mako.template import Template
block =  """
<head>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>
</head>

<%def name="figure(path, Title, control)">
<figure style="float:left;">
    % if control:
        <video src="${path}" id="master" controls />
    % else:
        <video src="${path}" class="slave" />
    %endif
        
    <figcaption> ${Title} </figcaption>
</figure>
</%def>
<% count = 0 %>

<h1 id="framecount">NAme</h1>
%for linmap in linmaps:
    % for tag in tags:
    <%
        gif = "%s/%s/recon.webm"%(tag, linmap) 
        if count == 0:
            control=True
        else:
            control =False
        count += 1
    %>
        ${figure(gif, gif, control)}
    %endfor
    <br />
% endfor

<script>
$(document).ready(function(){
  $("#master").on(
    "timeupdate", 
    function(event){
      updateOtherFrames(this.currentTime);
    });
});

function updateOtherFrames(time) {
    $(".slave").each( function(index, element){ this.currentTime = time; });
    frame = time * ${fps};
    $("h1").text("Frame " + frame.toString());
}
</script>
"""

block = """
<%def name="figure(path, Title)">
<figure style="float:left;">
        <img src="${path}" />
        
    <figcaption> ${Title} </figcaption>
</figure>
</%def>

%for img in imgs:
    ${figure(img, img)}
% endfor
"""


def get_data(tag, config):

    mod, *v = config['data'][tag]['obj']
    datamodule = importlib.import_module(mod)
    base = getattr(datamodule, v[0])


    if len(v) > 1:
        base = base[v[1]]

    return base


fps = 15
nframe =100
duration = nframe/fps

configfile: "recon.json"
workdir: "recons"

rule all:
    input: [recon.format(tag=tag) + '/recon.nc'\
               for recon in config['recons'].keys()\
               for tag in ['wthermo']]

rule clean:
    shell:"find . -name '*.nc' -exec rm -f {} \;"

rule contourvis:
    input: expand("{tag}/{linmap}/recon.nc", tag="wbin",\
                  linmap=config['recons'].keys())
    
    shell: "python -m rayben.gui.contour {input}"

rule concat:
    input: expand("wthermo/{linmap}/out.gif", linmap=config['recons'].keys())
    output: "wthermo/out.gif"
    run:
        from moviepy.editor import VideoFileClip, CompositeVideoClip
        vids = np.array([VideoFileClip(gif) for gif in input[:4]])
        vids.shape = (2,2)

        w, h = vids[0,0].size

        for i in range(vids.shape[0]):
            for j in range(vids.shape[1]):
                vids[i,j] = vids[i,j].set_pos((i*w, j*h))


        vid = CompositeVideoClip(list(vids.ravel()), size=(2*w, 2*h))
        vid.write_gif(output[0], fps=fps)

rule report:
    input: expand("{tag}/{linmap}/recon.webm", tag=config['tags'], linmap=config['recons'].keys())
    output: "index.html"
    run:
        linmaps = list(config['recons'].keys())
        tags    = config['tags']
        out = Template(block).render(tags=tags, linmaps=linmaps, fps=fps)


        with open("index.html", "w") as f:
            f.write(out)

        print(figures)

rule figs:
    input: "figs/uX-eig1.pdf", "figs/uX-eig11.pdf", "figs/uX-eig5-lag0.pdf", \
           "figs/uX-eig13-lag0.pdf", "figs/uX-eig12.pdf"
    run:
        from glob import glob

        imgs = glob("figs/*.png")

        out = Template(block).render(imgs=imgs)


        with open("index.html", "w") as f:
            f.write(out)

rule plot_alag:
    input: "{tag}/amat.nc"
    params: cmap='bwr'
    output: "figs/{tag}-eig{eig}-lag{lag}.pdf"
    run:
        eignum = int(wildcards.eig)
        lag    = int(wildcards.lag)
        xd = xray.open_dataset(input[0])['amat'].sel(eignum=eignum, lag=lag)
        if wildcards.tag == 'uX':
            xd = stm(xd)
        M = float(xd.max())

        fig,ax= plt.subplots(figsize=(3,2))
        ax.pcolormesh(*xargs(xd), vmin=-M, vmax=M, cmap=params.cmap, rasterized=True)
        ax.autoscale(tight=True)
        fig.savefig(output[0])

rule plot_alags:
    input: "{tag}/amat.nc"
    params: cmap='bwr'
    output: "figs/{tag}-eig{eig,\d+}.pdf"
    run:
        eignum = int(wildcards.eig)
        xd = xray.open_dataset(input[0])['amat'].sel(eignum=eignum)
        if wildcards.tag == 'uX':
            xd = stm(xd)

        M = float(xd.max())

        nlags = 20
        fig, axs = plt.subplots(5, 4, sharex=True, sharey=True)
        axs = axs.ravel()
        for i in range(nlags):
            lag = nlags-i-1
            z = xd.sel(lag=lag)


            ax = axs[i]
            im = axs[i].pcolormesh(*xargs(z), vmin=-M, vmax=M, cmap=params.cmap,
                                   rasterized=True)
            axs[i].autoscale(tight=True)
            ax.text(.1, 1.0, "Lag %2d"%lag, transform=ax.transAxes,
                    bbox=dict(fc="white"))

        cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
        plt.colorbar(im, cax=cax, **kw)
        fig.savefig(output[0])




# lag nc
rule composites:
    input: expand("figs/{linmap}-composite-roll{rolling}.pdf", linmap=config['recons'].keys(), rolling=[0,1])

rule composite:
    input: expand("{tag}/{{ linmap }}/recon.nc", tag=config['tags'])
    output: "figs/{linmap}-composite-roll{rolling}.pdf"
    run:
        from rayben.plots import plot_composites

        rolling = wildcards.rolling == '1'
        eigs = config['recons'][wildcards.linmap]['linmap']
        label = "Reconstruction composites with modes " + \
                ", ".join(map(str,eigs))

        plot_composites(wildcards.linmap, rolling=rolling, label=label,
                        figsize=(7,4))
        plt.savefig(output[0])
        

rule vis:
    input: "{tag}/{linmap}/recon.nc"
    output: "{tag}/{linmap}/recon.webm"
    run:
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        import xray

        fig, ax = plt.subplots(1,1,)
        z = xray.open_dataset(input[0])['w']
        
        # im = ax.imshow(z.values, cmap='spectral')

        M = max(-z.values.min(), z.values.max())
        levs = np.linspace(-M, M, 11)

        z0 = z.isel(t=0)
        xd, yd = z0.dims[::-1]
        x = z[xd].values
        y = z[yd].values

        im = ax.contourf(x, y, z.isel(t=0).values, levs, cmap='bwr')

        plt.colorbar(im)
        def update(t):
            t = t * fps
            ax.clear()
            ax.contourf(x, y, z.isel(t=int(t)).values, levs, cmap='bwr')
            # im.set_data(z.values)
            ax.set_title("Frame %d"%t)
            ax.set_xlabel(xd)
            ax.set_ylabel(yd)

            return mplfig_to_npimage(fig)
        
        update(0)

        clip = VideoClip(update, duration=duration).resize(.5)
        clip.write_videofile(output[0], fps=fps)

rule pair_plot:
    input: x="wthermo/{l1}/flux_pair.pkl", y="wthermo/{l2}/flux_pair.pkl"
    output: "figs/P{l1}-P{l2}.pdf"
    run:
        x = pd.read_pickle(input.x)
        y = pd.read_pickle(input.y)

        fig, axs= plt.subplots(2,2, sharex=True, sharey=True)
        for i in range(2):
            for j in range(2):
                axs[i,j].axhline(0.0, ls='--', c='k')
                axs[i,j].axvline(0.0, ls='--', c='k')
                axs[i,j].plot(y[i],x[j])
                axs[i,j].set_xlabel(input.x)
                axs[i,j].set_ylabel(input.y)
        plt.show()

rule phase_plot:
    input: "wthermo/{linmap}/flux_pair.pkl"
    output: "figs/P{linmap}.pdf"
    run:
        x = pd.read_pickle(input[0])
        x.plot(x=0, y=1)
        plt.show()

#------------------------------------------------------------
#              Reconstruction rules
#------------------------------------------------------------


def nc2dfpair(nc):

    wT = xray.open_dataset(nc)
    wT['half'] = wT.z >= .5

    arr = wT.groupby('half')\
            .apply(lambda x: integrate(x['wT'], axis='z'))
    df = pd.DataFrame(arr.values.T, index=arr.coords['t'])
    
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
        wT(xray.open_dataset(input[0])['w'])\
            .rename('wT')\
            .to_dataset()\
            .to_netcdf(output[0])

rule svd:
    input: "{dir}/amat.nc"
    output: a="{dir}/S{svdspec}/svd.pkl", o="{dir}/S{svdspec}/orthog.pkl"
    run:
        from numpy.linalg import svd
        amat = xray.open_dataset(input[0])['amat']


        spec = config['svds'][os.path.dirname(output[0])]
        inds = range(spec)

        U, S, V = svd(amat.values[:,inds], full_matrices=False)
        vT  = phi.ix[:,inds].dot(V.T)

        pickle.dump( (U, S, V, vT), open(output.a[0], "wb"))

        vT['metric'] = metric
        vT.to_pickle(output.o[0])


rule alags:
    input: expand("{{dir}}/{{tag}}/{lag}.amat.pkl", lag=range(config['lags']))
    output: "{dir}/{tag}/amat.nc"
    run:
        tag= wildcards.tag
        base = get_data(tag, config).isel(t=0)
        As = []
        lags = []

        for i in input:
            lag  = re.search("(\d+)\.amat\.pkl", i).group(1)
            lag  = int(lag)
            ind, A = pickle.load(open(i, "rb"))
            xd = df2xray(None, A, base, name='amat')\
                 .assign_coords(lag=lag)\
                 .rename({'t':'eignum'})

            As.append(xd)

        xray.concat(As, 'lag').to_dataset().to_netcdf(output[0])
        


rule pkl2nc:
    input: "{dir}/{tag}/R{recon}/recon.pkl"
    output: "{dir}/{tag}/R{recon}/recon.nc"
    run:
        tag= wildcards.tag
        base = get_data(tag, config).isel(t=0)
        df = pd.read_pickle(input[0])
        df2xray(df.index, df.values, base, name='w')\
            .to_dataset()\
            .to_netcdf(output[0])


def get_recon(tag, rdir, config):
    for k in config['recons']:
        if rdir == k.format(tag=tag):
            print(k.format(tag=tag))
            return config['recons'][k]

rule recon_all:
    input: a=expand("{{dir}}/{{tag}}/{lag}.amat.pkl", lag=range(config['lags'])),\
           o="{dir}/orthog.pkl"
    output: pkl="{dir}/{tag}/R{recon}/recon.pkl"
    run:
        pkl = output.pkl[0]
        rc = get_recon(wildcards.tag, os.path.dirname(pkl), config)
        recon_all(input.a, input.o[0], pkl, **rc)



rule split:
    input: "{orthog}/orthog.pkl"
    output: "{orthog}/{tag}/{lag}.amat.pkl"
    run:
        try:
            os.mkdir(wildcards.tag)
        except:
            pass
        tag = wildcards.tag
        base = get_data(tag, config)
        phi  = pd.read_pickle(input[0])
        mk_amat(output[0], base, int(wildcards.lag), phi, field='w')

rule phi:
    output: "alpha0/orthog.pkl"
    run:
        phi['metric'] = metric
        phi.to_pickle(output[0])

