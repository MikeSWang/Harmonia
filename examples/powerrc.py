"""Power spectrum recovery runtime configuration.

This sets I/O paths and provides common parameters and functionalities to power
spectrum recovery scripts.

"""
import os
from sys import argv, path
from argparse import ArgumentParser

import numpy


def get_filename(*filepath):
    if not filepath:
        return os.path.splitext(os.path.basename(argv[0]))[0]
    return os.path.splitext(os.path.basename(filepath[0]))[0]


def save_data(dirpath, file, data):
    if not dirpath.endswith("/"):
        dirpath += "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    numpy.save("".join([dirpath, file]), data)


PATHIN = "./data/input/"
PATHOUT = "./data/output/"

path.insert(0, "../")

fname = get_filename()
fdir = "{}/".format(fname)

parser = ArgumentParser(description="Power spectrum recovery set-up.")

parser.add_argument('--nbar', type=float, default=1e-3)
parser.add_argument('--contrast', type=float, default=None)
parser.add_argument('--zmax', type=float, default=0.05)
parser.add_argument('--kmax', type=float, default=0.1)
parser.add_argument('--dk', type=float, default=1e-2)

parser.add_argument('--boxside', type=float, default=1000.)
parser.add_argument('--expand', type=float, default=2.)
parser.add_argument('--meshgen', type=int, default=256)
parser.add_argument('--meshcal', type=int, default=256)

parser.add_argument('--infile', type=str, default="halos-(NG=0.,z=1.)-0L.txt")
parser.add_argument('--niter', type=int, default=25)
parser.add_argument('--progid', type=str, default="")
