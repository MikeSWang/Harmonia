"""Field generator runtime configuration.

This sets I/O paths and provides common parameters and functionalities to field
fidelity assurance scripts.

"""
import os
import sys
import warnings
from argparse import ArgumentParser

sys.path.insert(0, "../")

from harmonia.collections import collate, harmony

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def clean_warnings(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def collate_and_save(pattern, extension, save=True, opath=None, oname=None):

    def owck(path_tocheck):
        if os.path.exists(path_tocheck):
            raise FileExistsError

    output, count, _ = collate(pattern, extension)

    overwrite_permission = False
    while save:
        import numpy
        try:
            assert confirm_dir(opath)
            if not overwrite_permission:
                owck(opath + oname)
            numpy.save(opath + oname, output)
        except FileExistsError:
            assent = input(
                "Saving would overwrite existing file at destination. "
                "Do you want to continue? [y/n] "
                )
            if assent.lower().startswith('y'):
                overwrite_permission = True
                continue
            else:
                print("Overwrite permission denied. File not saved. ")
        break

    return output, count


def confirm_dir(dirpath):
    if not dirpath.endswith("/"):
        dirpath += "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return os.path.exists(dirpath)


def get_filename(*filepath):
    if not filepath:
        filepath = [sys.argv[0]]
    return os.path.splitext(os.path.basename(filepath[0]))[0]


def parse_cli_args(cli_parser):

    cli_parser.add_argument('--stat', default='lognormal')
    cli_parser.add_argument('--noncat', action='store_true')
    cli_parser.add_argument('--nosmp', action='store_true')

    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0.)

    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=None)
    cli_parser.add_argument('--boxside', type=float, default=1000.)
    cli_parser.add_argument('--bins', type=int, default=15)
    cli_parser.add_argument('--meshgen', type=int, default=256)
    cli_parser.add_argument('--meshcal', type=int, default=256)

    cli_parser.add_argument('--niter', type=int, default=25)
    cli_parser.add_argument('--progid', default="")

    return cli_parser.parse_args()


# Warning reformatting
warnings.formatwarning = clean_warnings

# I/O paths and files
fname = get_filename()
fdir = "{}/".format(fname)

# Command-line inputs
parser = ArgumentParser(description="Field fidelity assurance set-up.")
params = parse_cli_args(parser)
