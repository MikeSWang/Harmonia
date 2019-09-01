"""Power spectrum recovery runtime configuration.

This sets I/O paths and provides common parameters and functionalities to power
spectrum recovery scripts.

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
    cli_parser.add_argument('--zmax', type=float, default=0.05)
    cli_parser.add_argument('--rmax', type=float, default=150.)
    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=1e-2)

    # Computing parameters
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
    return {
        'Nk': np.sum(result['Nk'], axis=0),
        'k': np.average(result['k'], axis=0),
        'Pk': np.average(result['Pk'], axis=0),
        'Pshot': np.average(result['Pshot']),
        'ln': np.atleast_2d(result['ln']),
        'kln': np.atleast_2d(result['kln'])[0],
        'Pln': np.average(result['Pln'], axis=0),
        'dk': np.std(result['k'], axis=0, ddof=1),
        'dPk': np.std(result['Pk'], axis=0, ddof=1),
        'dPln': np.std(result['Pln'], axis=0, ddof=1),
        'dof1': np.size(result['k'], axis=-1) - 1,
        'dof2': np.size(result['Pln'], axis=0) - 1,
        }


# I/O paths and files
PATHIN = "./data/input/"
PATHOUT = "./data/output/"

path.insert(0, "../")

fname = get_filename()
fdir = "{}/".format(fname)

# Command-line inputs
parser = ArgumentParser(description="Power spectrum recovery set-up.")
params = parse_cli_args(parser)
