"""Runtime configuration for map processing.

"""
import logging
import os
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../../")))

from harmonia.collections import cartesian_to_spherical
from harmonia.collections import clean_warning_format, get_filename


def parse_external_args():
    """Parse command line arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameters.

    """
    parser = ArgumentParser()

    parser.add_argument('--map', required=True)
    parser.add_argument('--fsky', type=float, default=1.)
    parser.add_argument('--kmin', type=float, default=None)
    parser.add_argument('--kmax', type=float, default=0.1)
    parser.add_argument('--rsd', action='store_true')
    parser.add_argument('--multipoles', type=int, nargs='+', default=0)

    parser.add_argument('--nbar', type=float, default=2.4883e-4)
    parser.add_argument('--contrast', type=float, default=10.)
    parser.add_argument('--boxsize', type=float, default=1000.)
    parser.add_argument('--dk', type=float, default=None)
    parser.add_argument('--mesh', type=int, default=256)
    parser.add_argument('--input-catalogue', required=True)

    return parser.parse_args()


def domain_cut(cartesian_position, radius, fraction, split_caps=False):
    """Define the sky domain cut.

    Parameters
    ----------
    cartesian_position : float array_like
        Cartesian positions.
    radius : float
        Domain radius.
    fraction : float
        Sky fraction.
    split : bool, optional
        If `True` (default is `False`), the sky mask is split between north
        and south polar caps in antipodal regions.

    Returns
    -------
    veto : bool :class:`numpy.ndarray`
        Domain veto value.

    """
    spherical_position = cartesian_to_spherical(
        cartesian_position - [radius] * 3
    )

    if split_caps:
        veto = np.logical_or(
            np.logical_and(
                spherical_position[:, 1] < np.pi/2,
                spherical_position[:, 2] <= fraction * (2*np.pi)
            ),
            np.logical_and(
                spherical_position[:, 1] >= np.pi/2,
                spherical_position[:, 2] >= (1 - fraction) * (2*np.pi)
            )
        )
    else:
        veto = spherical_position[:, 2] <= fraction * (2*np.pi)

    return veto


warnings.formatwarning = clean_warning_format

PATHIN = Path("../../data/survey/")
PATHOUT = Path("../../data/map/")

if __name__ != '__main__':
    script_name = get_filename(sys.argv[0])

    logging_handler = logging.StreamHandler(sys.stdout)
    logging_formatter = logging.Formatter(
        fmt='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging_handler.setFormatter(logging_formatter)

    logger = logging.getLogger()
    logger.addHandler(logging_handler)
    logger.setLevel(logging.INFO)