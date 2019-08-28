"""Power spectrum recovery runtime configuration.

This sets file I/O paths and passes command line arguments to scripts.  It also
provides the function to extract file name from the script being run.

"""
from sys import argv, path
from os.path import basename, splitext

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def get_filename(filepath):
    return splitext(basename(filepath))[0]


path.insert(0, "../")

fname = get_filename(argv[0])
fdir = "{}/".format(fname)
