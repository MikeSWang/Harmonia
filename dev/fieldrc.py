"""Field generator runtime configuration.

This sets I/O paths and provides common parameters and functionalities to
field fidelity assurance scripts.

"""
import os
import warnings
from argparse import ArgumentParser
from sys import argv, path

import numpy as np

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def get_filename(*filepath):
    if not filepath:
        filepath = [argv[0]]
    return os.path.splitext(os.path.basename(filepath[0]))[0]


def confirm_dir(dirpath):
    if not dirpath.endswith("/"):
        dirpath += "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return os.path.exists(dirpath)


def clean_warnings(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def parse_cli_args(cli_parser):

    cli_parser.add_argument('--stat', default='lognormal')
    cli_parser.add_argument('--smp', action='store_true')

    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0)

    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--boxside', type=float, default=1000.)
    cli_parser.add_argument('--meshgen', type=int, default=256)
    cli_parser.add_argument('--meshcal', type=int, default=256)

    cli_parser.add_argument('--niter', type=int, default=25)
    cli_parser.add_argument('--progid', default="")

    return cli_parser.parse_args()


def aggregate(result):
    return {var: np.average(val, axis=0) for var, val in result.items()}


path.insert(0, "../")
warnings.formatwarning = clean_warnings

# I/O paths and files
fname = get_filename()
fdir = "{}/".format(fname)

# Command-line inputs
parser = ArgumentParser(description="Field fidelity assurance set-up.")
params = parse_cli_args(parser)
