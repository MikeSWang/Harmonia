"""Two-point recovery runtime configuration.

This sets file I/O paths and passes command line arguments to scripts.  It also
provides the function to extract file name from the script being run.

"""
import os
from sys import argv, path

import numpy

PATHIN = "./data/input/"
PATHOUT = "./data/output/"


def get_filename(*filepath):
    if not filepath:
        return os.path.splitext(os.path.basename(argv[0]))[0]
    return os.path.splitext(os.path.basename(filepath[0]))[0]


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(path, data):
    create_directory(path)
    numpy.save(path, data)


path.insert(0, "../")

fname = get_filename()
fdir = "{}/".format(fname)
