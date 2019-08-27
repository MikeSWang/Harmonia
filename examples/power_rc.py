"""Power spectrum recovery runtime configuration.

This sets file I/O paths and passes command line arguments to scripts.  It also
provides the function to extract file name from the script being run.

"""
import sys
from argparse import ArgumentParser
from os.path import basename, splitext


def get_filename(filepath):
    return splitext(basename(filepath[0]))[0]


# -- SYSTEM PATHS -------------------------------------------------------------

PATHIN = "./data/input/"
PATHOUT = "./data/output/"

fname = get_filename(sys.argv)
fdir = "{}/".format(fname)

sys.path.insert(0, "../")


# -- SYSTEM ARGUMENTS ---------------------------------------------------------

parser = ArgumentParser(description="Power spectrum recovery set-up. ")

# Field properties
parser.add_argument('--nbar', type=float, default=1e-3)
parser.add_argument('--bias', type=float, default=2.)
parser.add_argument('--redshift', type=float, default=0.)

# Spectrum properties
parser.add_argument('--kmax', type=float, default=2.)
parser.add_argument('--dk', type=float, default=1e-2)

# Catalogue properties
parser.add_argument('--zmax', type=float, default=0.05)
parser.add_argument('--rmax', type=float, default=150.)
parser.add_argument('--boxside', type=float, default=1000.)
parser.add_argument('--expand', type=float, default=2.)
parser.add_argument('--contrast', type=float, default=None)
parser.add_argument('--infile', type=str, default="halos-(NG=0.,z=1.)-0L.txt")

# Algorithm options
parser.add_argument('--meshgen', type=int, default=256)
parser.add_argument('--meshcal', type=int, default=256)
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--progid', type=str, default='')

params = parser.parse_args()
