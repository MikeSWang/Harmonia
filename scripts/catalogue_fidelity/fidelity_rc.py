"""Runtime configuration for catalogue fidelity tests.

"""
import os
import sys
import warnings
from argparse import ArgumentParser

PATHIN = "../../data/catalogue_fidelity/input/"
PATHOUT = "../../data/catalogue_fidelity/output/"


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

    cli_parser.add_argument('--generator', default='lognormal')
    cli_parser.add_argument('--non-catalogue', action='store_true')
    cli_parser.add_argument('--no-sampling', action='store_true')

    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0.)

    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=None)
    cli_parser.add_argument('--nbins', type=int, default=15)

    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--mesh-gen', type=int, default=256)
    cli_parser.add_argument('--mesh-cal', type=int, default=256)

    cli_parser.add_argument('--niter', type=int, default=10)
    cli_parser.add_argument('--prog-id', default="")

    return cli_parser.parse_args()


if not __name__ == '__main__':

    import_local_package()

    from harmonia.collections import clean_warning_format, get_filename

    warnings.formatwarning = clean_warning_format
    script_name = get_filename(sys.argv[0])
    params = parse_cli_args()
