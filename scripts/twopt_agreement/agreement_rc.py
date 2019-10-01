"""Two-point function agreement runtime configuration.

"""
import os
import sys
import warnings
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def clean_warnings(message, category, filename, lineno, line=None):
    """Clean warning message format.

    Parameters
    ----------
    message, category, filename, lineno : str
        Warning message, warning catagory, origin file name, line number.
    line : str or None, optional
        Source code line to be included in the warning message (default is
        `None`).

    Returns
    -------
    str
        Warning message format.

    """
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def import_local_package():
    """Add package to Python module path.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../../")))


def parse_cli_args():
    """Parse command line arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameters.

    """
    cli_parser = ArgumentParser()

    cli_parser.add_argument('--struct')
    cli_parser.add_argument('--rsd', action='store_true')

    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--contrast', type=float, default=None)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0.)

    cli_parser.add_argument('--zmax', type=float, default=0.05)
    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=1e-2)

    cli_parser.add_argument('--boxside', type=float, default=1000.)
    cli_parser.add_argument('--expand', type=float, default=2.)
    cli_parser.add_argument('--mesh-gen', type=int, default=256)
    cli_parser.add_argument('--mesh-cal', type=int, default=256)

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


def average(result):
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


if not __name__ == '__main__':

    warnings.formatwarning = clean_warnings

    import_local_package()

    from harmonia.collections import get_filename

    script_name = get_filename(sys.argv[0])
    params = parse_cli_args()

