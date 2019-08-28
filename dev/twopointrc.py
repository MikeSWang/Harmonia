"""Two-point recovery runtime configuration.

This sets file I/O paths and passes command line arguments to scripts.  It also
provides the function to extract file name from the script being run.

"""
import os
from sys import argv, path

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def get_filename(*filepath):
    if not filepath:
        return os.path.splitext(os.path.basename(argv[0]))[0]
    return os.path.splitext(os.path.basename(filepath))[0]


path.insert(0, "../")

fname = get_filename()
fdir = f"{fname}/"

if not os.path.exists(PATHIN):
    os.makedirs(PATHIN)
if not os.path.exists(PATHOUT + fdir):
    os.makedirs(PATHOUT + fdir)
