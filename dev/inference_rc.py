"""Likelihood inference runtime configuration.

This sets I/O paths and provides common parameters and functionalities to
likelihood inference scripts.

"""
import os
import sys
import warnings
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def get_filename(*filepath):
    if not filepath:
        filepath = [sys.argv[0]]
    return os.path.splitext(os.path.basename(filepath[0]))[0]


def parse_cli_args(cli_parser):
    # Physical parameters
    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--contrast', type=float, default=None)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0.)
    cli_parser.add_argument('--rsd', action='store_true')

    cli_parser.add_argument('--zmax', type=float, default=0.05)
    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=1e-2)

    # Computing parameters
    cli_parser.add_argument('--struct')
    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--expand', type=float, default=2.)
    cli_parser.add_argument('--meshgen', type=int, default=256)
    cli_parser.add_argument('--meshcal', type=int, default=256)

    # Program parameters
    cli_parser.add_argument('--niter', type=int, default=25)
    cli_parser.add_argument('--progid', default="")

    return cli_parser.parse_args()


def clean_warnings(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def confirm_dir(dirpath):
    if not dirpath.endswith("/"):
        dirpath += "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def aggregate(result):
    return {var: np.average(val, axis=0) for var, val in result.items()}


def mpicomp(data_arr, mappings, comm, root=0):

    from harmonia.collections import allocate_segments

    segment = allocate_segments(ntask=len(data_arr), nproc=comm.size)
    data_chunk = data_arr[segment[comm.rank]]

    outputs = defaultdict(list)
    for var, comp in mappings.items():
        for data_element in data_chunk:
            outputs[var].append(comp(data_element))

    comm.Barrier()

    result = {var: comm.gather(val, root=root) for var, val in outputs.items()}

    if comm.rank == root:
        result = {
            var: np.concatenate(val, axis=0) for var, val in result.items()
            }

    return result


sys.path.insert(0, "../../")
warnings.formatwarning = clean_warnings

filename = get_filename()

params = parse_cli_args(ArgumentParser())
