"""Validate spherical modelling.

"""
import os
import sys
from collections import defaultdict
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpi4py import MPI

try:
    from application import confirm_directory, data_dir
    from harmonia.cosmology import BaseModel
    from harmonia.reader import Couplings, SphericalCorrelator
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import confirm_directory, data_dir
    from harmonia.cosmology import BaseModel
    from harmonia.reader import Couplings, SphericalCorrelator


def validate_full_sky_spherical_model():
    """Validate full sky spherical modelling.

    """
    couplings = Couplings.load(survey_product_dir/couplings_file)

REDSHIFT = 1.
NG = 0

if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    # Set I/O paths.
    survey_product_dir = data_dir/"processed"/"survey_products"

    couplings_info = ",".join([
        "kmax={}".format(0.04),
        "mask={}".format("random0_BOSS_DR12v5_CMASS_North"),
        "selection={}".format("[100.0,500.0]")
    ])

    couplings_file = "couplings-({}).npz".format(couplings_info)

    cosmo_dir = data_dir/"external"/"cosmology"

    cosmo_file = "simulation.txt"

    output_dir = data_dir/"processed"/"survey_validation"

    # Validate spherical modelling.
    confirm_directory(output_dir)
