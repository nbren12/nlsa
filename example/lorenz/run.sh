rm -rf u
snakemake lorenz96.nc u/q20/Ebase/orthog.pkl
python ../../plot_phi.py  u/q20/Ebase/eigs.pkl
