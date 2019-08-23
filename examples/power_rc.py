import sys.path
sys.path.insert(0, "../")

from argparse import ArgumentParser
from os.path import basename, splitext
from sys import argv

from harmonia.collections import format_float as ff, harmony


def get_filename(filepath):
    return splitext(basename(filepath[0]))[0]


# -- SYSTEM PATHS -------------------------------------------------------------

PATHIN = "./data/input/"
PATHOUT = "./data/output/"

fname = get_filename(argv)
fdir = "{}/".format(fname)


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
