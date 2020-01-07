"""Runtime configuration for likelihood sampling.

"""
import os
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

PATHIN = Path("../../data/survey/")
PATHOUT = Path("../../data/likelihood/")


def import_local_package():
    """Add package to Python module path.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../../")))


def parse_external_args():
    """Parse command line arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameters.

    """
    parser = ArgumentParser()

    parser.add_argument('--breakdown', action='store_true')
    parser.add_argument('--spherical-pivot', default='spectral')
    parser.add_argument('--cartesian-pivot', default='order')

    parser.add_argument('--map', required=True)
    parser.add_argument('--khyb', type=float, default=0.04)
    parser.add_argument('--kmax', type=float, default=0.1)
    parser.add_argument('--multipoles', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--num-cov-est', type=int, default=None)

    parser.add_argument('--nbar', type=float, default=2.4883e-4)
    parser.add_argument('--growth-rate', type=float, default=0.)
    parser.add_argument('--redshift', type=float, default=1.)

    parser.add_argument('--contrast', type=float, default=10.)
    parser.add_argument('--boxsize', type=float, default=1000.)
    parser.add_argument('--dk', type=float, default=None)
    parser.add_argument('--mesh', type=int, default=256)

    parser.add_argument('--input-catalogue', required=True)
    parser.add_argument('--load-couplings', action='store_true')

    return parser.parse_args()


if __name__ != '__main__':

    import_local_package()

    from harmonia.collections import clean_warning_format, get_filename

    warnings.formatwarning = clean_warning_format

    script_name = get_filename(sys.argv[0])
