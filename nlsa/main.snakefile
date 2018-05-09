# vim: syntax=python tabstop=4 expandtab
"""
An Example workflow for NLSA analysis on a rayleigh benard convection dataset

Usage:
1. Specify path to NLSA root
2. Specify the configuration file
3. Run `snakemake <target>` from the shell

"""
import os
import sys
import nlsa

# specify path to NLSA
nlsapath = nlsa.__path__[0]

#
#  Include snakefiles
#

include: os.path.join(nlsapath, "recon.snakefile")
include: os.path.join(nlsapath, "diffusionmap.snakefile")
