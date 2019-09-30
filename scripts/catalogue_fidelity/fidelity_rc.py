"""Runtime configuration for catalogue fidelity tests.

"""
import os
import sys
import warnings
from argparse import ArgumentParser

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


def overwrite_protection(outpath, outname, save=True):
    """Inspect and modify overwrite protection.

    Parameters
    ----------
    outpath : str
        Write-out directory path.
    outname : str
        Write-out filename.

    Returns
    -------
    overwrite_permission : bool
        Overwrite permission.

    """
    overwrite_permission = False
    while save:
        try:
            if not os.path.exists(outpath):
                raise FileNotFoundError(f"{outpath} does not exist. ")
            if not overwrite_permission:
                if os.path.exists(outpath + outname):
                    raise FileExistsError
            overwrite_permission = True
            break
        except FileExistsError:
            grant_permission = input(
                "Saving would overwrite existing file at destination. "
                "Do you want to continue? [y/n] "
            )
            if grant_permission.lower().startswith('y'):
                overwrite_permission = True
                break
            else:
                overwrite_permission = False
                raise FileExistsError(
                    "Overwrite permission denied. File not saved. "
                    )

    return overwrite_permission


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

    cli_parser.add_argument('--niter', type=int, default=25)
    cli_parser.add_argument('--prog-id', default="")

    return cli_parser.parse_args()


def _view


if not __name__ == '__main__':

    warnings.formatwarning = clean_warnings

    import_local_package()

    from harmonia.collections import get_filename

    script_name = get_filename(sys.argv[0])
    params = parse_cli_args()
