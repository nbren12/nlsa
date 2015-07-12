configfile: "nlsa.yaml"
include: "nlsa/diffusionmap.snakefile"

workdir: "anl"

rule all:
    input: "wthermo/q20/e1_a0/eigs.pkl"

