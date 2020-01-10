"""Determine the survey window function.

"""
from argparse import ArgumentParser
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from window_rc import PATHOUT
from harmonia.mapper import SurveyWindow
from harmonia.collections import (
    cartesian_to_spherical,
    confirm_directory_path,
    harmony,
    spherical_indicator,
)

plt.style.use(harmony)
sns.set(style='ticks', font='serif')


def parse_cli_args():
    """Parse command-line argument inputs.

    """
    cli_parser = ArgumentParser()

    cli_parser.add_argument('--fsky', type=float, default=1.)
    cli_parser.add_argument('--split', action='store_true')
    cli_parser.add_argument('--nbar', type=float, default=1e-2)
    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--padding', type=float, default=70.)
    cli_parser.add_argument('--mesh', type=int, default=512)

    return cli_parser.parse_args()


def sky_mask(cartesian_position, fraction, split_caps=False):
    """Define the sky mask.

    Parameters
    ----------
    cartesian_position : float array_like
        Cartesian positions.
    fraction : float
        Sky fraction.
    split : bool, optional
        If `True` (default is `False`), the mask is split between north and
        south polar caps in antipodal regions.

    Returns
    -------
    mask : bool :class:`numpy.ndarray`
        Sky mask.

    """
    spherical_position = cartesian_to_spherical(cartesian_position)

    if split_caps:
        mask = np.logical_or(
            np.logical_and(
                spherical_position[:, -1] <= fraction * (2*np.pi),
                spherical_position[:, 1] < np.pi/2
            ),
            np.logical_and(
                spherical_position[:, -1] >= (1 - fraction) * (2*np.pi),
                spherical_position[:, 1] >= np.pi/2
            )
        )
    else:
        mask = spherical_position[:, -1] <= fraction * (2*np.pi)

    return mask


def synthesise():
    """Synthesise a high-density random catalogue with the survey window.

    Returns
    -------
    :class:`harmonia.mapper.window_synthesis.SurveyWindow`
        Survey window.

    """
    _window = SurveyWindow(
        mask=lambda pos: \
            spherical_indicator(pos, boxsize/2) \
            * sky_mask(pos, fsky, split_caps=split)
    )
    _window.synthesise(nbar, boxsize, padding=padding)

    return _window


def determine_window():
    """Determine window multipoles from the synthetic catalogue.

    Returns
    -------
    tuple of dict
        Survey window correlation and power multipoles.

    """
    _xi_ell = window.correlation_function_multipoles(ORDERS, num_mesh=mesh)
    _pk_ell = window.power_multipoles

    return _xi_ell, _pk_ell


if __name__ == '__main__':

    confirm_directory_path(PATHOUT/"window_multipoles")

    params = parse_cli_args()
    pprint(params.__dict__)

    fsky = params.fsky
    split = params.split
    nbar = params.nbar
    boxsize = params.boxsize
    padding = params.padding
    mesh = params.mesh

    ORDERS = [0, 2, 4, 6, 8]

    window = synthesise()
    xi_ell, pk_ell = determine_window()

    np.save(
        PATHOUT/"window_multipoles"/(
            f"mask_multipoles-{{:.2f}}sky{{}}-{{:.0f}}pad-{{}}mesh.npy"
            .format(fsky, split*"-split", padding, mesh)
        ),
        xi_ell
    )
    np.save(
        PATHOUT/"window_multipoles"/(
            f"window_multipoles-{{:.2f}}sky{{}}-{{:.0f}}pad-{{}}mesh.npy"
            .format(fsky, split*"-split", padding, mesh)
        ),
        pk_ell
    )

    plt.close('all')
    plt.figure(figsize=(8, 11))

    plt.subplot2grid((2, 1), (0, 0))
    for ell in ORDERS:
        plt.semilogx(
            pk_ell['k'], pk_ell[f'power_{ell}'],
            label=r'$\ell={}$'.format(ell)
        )
    plt.xlabel(r"$k$")
    plt.ylabel(r"$Q_\ell(k)$")
    plt.legend()

    plt.subplot2grid((2, 1), (1, 0))
    for ell in ORDERS:
        plt.semilogx(
            xi_ell['s'], xi_ell[f'correlation_{ell}'],
            label=r'$\ell={}$'.format(ell)
        )
    plt.xlabel(r"$s$")
    plt.ylabel(r"$Q_\ell(s)$")
    plt.legend()

    plt.savefig(
        PATHOUT/"window_multipoles"/(
            f"window_multipoles-{{:.2f}}sky{{}}-{{:.0f}}pad-{{}}mesh.pdf"
            .format(fsky, split*"-split", padding, mesh)
        )
    )
