"""Two-point recovery runtime configuration.

This sets I/O paths and provides common parameters and functionalities to
2-point function recovery scripts.

"""
import os
from argparse import ArgumentParser
from sys import argv, path

import numpy as np


def get_filename(*filepath):
    if not filepath:
        filepath = [argv[0]]
    return os.path.splitext(os.path.basename(filepath[0]))[0]


def parse_cli_args(cli_parser):
    # Physical parameters
    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--contrast', type=float, default=None)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0)
    cli_parser.add_argument('--rsd', action='store_true')

    cli_parser.add_argument('--zmax', type=float, default=0.05)
    cli_parser.add_argument('--rmax', type=float, default=150.)
    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=1e-2)

    # Computing parameters
    cli_parser.add_argument('--struct', required=True)
    cli_parser.add_argument('--boxside', type=float, default=1000.)
    cli_parser.add_argument('--expand', type=float, default=2.)
    cli_parser.add_argument('--meshgen', type=int, default=256)
    cli_parser.add_argument('--meshcal', type=int, default=256)

    # Program parameters
    cli_parser.add_argument('--niter', type=int, default=25)
    cli_parser.add_argument('--progid', default="")
    cli_parser.add_argument('--infile', default="halos-(NG=0.,z=1.)-0L.txt")

    return cli_parser.parse_args()


def confirm_dir(dirpath):
    if not dirpath.endswith("/"):
        dirpath += "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def aggregate(result):
    return {var: np.average(val, axis=0) for var, val in result.items()}


# I/O paths and files
PATHIN = "./data/input/"
PATHOUT = "./data/output/"

path.insert(0, "../")

fname = get_filename()
fdir = "{}/".format(fname)

# Command-line inputs
parser = ArgumentParser(description="Two-point function recovery set-up.")
params = parse_cli_args(parser)
