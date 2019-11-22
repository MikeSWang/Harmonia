"""Runtime configuration for likelihood sampling.

"""
import os
import sys
import warnings
from argparse import ArgumentParser

PATHIN = "./data/input/"
PATHOUT = "./data/output/"
# TODO: "../../data/inference/input/", "../../data/inference/output/"


def import_local_package():
    """Add package to Python module path.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))
    # TODO: "../../"


def parse_cli_args():
    """Parse command line arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameters.

    """
    cli_parser = ArgumentParser()

    cli_parser.add_argument('--map', default='hybrid')

    cli_parser.add_argument('--sample-param', required=True)
    cli_parser.add_argument('--fixed-param', required=False)
    cli_parser.add_argument('--prior-range', type=float, nargs=2)
    cli_parser.add_argument('--num-sample', type=int, default=100)

    cli_parser.add_argument('--ksplit', type=float, default=0.05)
    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--breakdown', action='store_true')

    cli_parser.add_argument('--spherical-pivot', default='spectral')
    cli_parser.add_argument('--cartesian-pivot', default='multipole')
    cli_parser.add_argument('--load-couplings', action='store_true')

    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--bias', type=float, nargs=2, default=[2., 2.])
    cli_parser.add_argument('--fnl', type=float, default=None)
    cli_parser.add_argument('--growth-rate', type=float, default=None)
    cli_parser.add_argument('--redshift', type=float, default=0.)

    cli_parser.add_argument('--contrast', type=float, default=10.)
    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--dk', type=float, default=None)
    cli_parser.add_argument('--mesh', type=int, default=256)

    cli_parser.add_argument('--prog-id', required=False)
    cli_parser.add_argument('--input-file', required=False)

    return cli_parser.parse_args()


if __name__ != '__main__':

    import_local_package()

    from harmonia.collections import clean_warning_format, get_filename

    warnings.formatwarning = clean_warning_format

    script_name = get_filename(sys.argv[0])

    params = parse_cli_args()
