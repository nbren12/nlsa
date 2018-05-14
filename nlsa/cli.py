import os
import subprocess
import argparse
import tempfile

example_config = """
"""
description = """Run the NLSA algorithm given a configuration file

This file should have the following contents:

    data:
    u:
        netcdf: lorenz96.nc
        var: u
        tdim: time
    diffmaps:
    base:
        eps: 1.0
        alpha: 0.5
        symmetric_eigs: false
        n: 40
        sparsity: .01
    svds:
    "40": 40
    tags:
    - u

"""


config_template = """
data:
  {variable_name}:
    netcdf: {input_netcdf}
    var: {variable_name}
    tdim: time
diffmaps:
  cli:
    eps: {epsilon}
    alpha: {alpha}
    symmetric_eigs: false
    n: {n}
    sparsity: {sparsity}
"""


def make_config_file(input_netcdf, variable_name, alpha, epsilon, n, sparsity):
    return config_template.format(**locals())


def main():
    mydir = os.path.abspath(os.path.dirname(__file__))
    snakefile = os.path.join(mydir, "main.snakefile")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("-i", "--input", type=str, help="Input netcdf")
    parser.add_argument("-v", "--variable", type=str, help="Variable name")
    parser.add_argument("-a", "--alpha", type=float, default=1.0,
                        help="Coifman and Lafon parameter")
    parser.add_argument("-e", "--epsilon", type=float, default=1.0,
                        help="Bandwidth of kernel")
    parser.add_argument("-q", type=int, default=100,
                        help="Embedding window length")
    parser.add_argument("-n", type=int, default=100,
                        help="Number of eigenfunctions to find")
    parser.add_argument("-s", "--sparsity", type=float, default=.01,
                        help="Sparsity of pairwise distance map")

    args = parser.parse_args()

    if args.config:
        config_path = args.config
        eig_path = "u/q20/Ecli/orthog.pkl"
    else:
        # open and write config file
        config = make_config_file(args.input, args.variable, args.alpha,
                                  args.epsilon, args.n, args.sparsity)
        config_path = tempfile.mktemp()
        with open(config_path, "w") as f:
            f.write(config)

        # get the orthog name
        eig_path = f"{args.variable}/q{args.q}/Ecli/orthog.pkl"

    subprocess.call([
        "snakemake",
        "--configfile", config_path,
        "-s", snakefile,
        eig_path
    ])
