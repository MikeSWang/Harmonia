"""Runtime configuration for likelihood sampling.

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

from harmonia.collections import (
    cartesian_to_spherical,
    clean_warning_format,
    get_filename,
    setup_logger,
)


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
    parser.add_argument('--khyb', type=float, default=0.04)
    parser.add_argument('--kmax', type=float, default=0.1)
    parser.add_argument('--rsd', action='store_true')
    parser.add_argument('--multipoles', type=int, nargs='+', default=[0])

    parser.add_argument('--spherical-pivot', default='spectral')
    parser.add_argument('--cartesian-pivot', default='order')
    parser.add_argument('--breakdown', action='store_true')
    parser.add_argument('--load-couplings', action='store_true')
    parser.add_argument('--num-cov-est', type=int, default=None)

    parser.add_argument('--nbar', type=float, default=2.4883e-4)
    parser.add_argument('--contrast', type=float, default=10.)
    parser.add_argument('--redshift', type=float, default=1.)
    parser.add_argument('--boxsize', type=float, default=1000.)
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
PATHOUT = Path("../../data/likelihood/")
DATAPATH = Path("../../data/map/")

if __name__ != '__main__':
    script_name = get_filename(sys.argv[0])
    logger = setup_logger()
    logger.setLevel(logging.INFO)
