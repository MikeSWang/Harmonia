"""Calculate the survey window function.

"""
import os
import sys
from argparse import ArgumentParser
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

from harmonia.mapper import SurveyWindow
from harmonia.collections import (
    cartesian_to_spherical,
    harmony,
    spherical_indicator,
)

plt.style.use(harmony)
sns.set(style='ticks', font='serif')


def parse_cli_args():
    """Parse command-line argument inputs.

    """
    cli_parser = ArgumentParser()

    cli_parser.add_argument('--fsky', type=float, default=1/3)
    cli_parser.add_argument('--nbar', type=float, default=2.4883e-4)
    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--padding', type=float, default=70.)

    return cli_parser.parse_args()


def sky_mask(cartesian_position, fraction):
    """Define the sky mask.

    Parameters
    ----------
    cartesian_position : float array_like
        Cartesian positions.
    fraction : float
        Sky fraction.

    Returns
    -------
    mask: bool :class:`numpy.ndarray`
        Sky mask.

    """
    spherical_position = cartesian_to_spherical(cartesian_position)

    mask = spherical_position[..., -1] <= fraction * (2*np.pi)

    return mask


def synthesise():
    """Synthesise a high-density random catalogue with the surveywindow.

    Returns
    -------
    :class:`harmonia.mapper.window_synthesis.SurveyWindow`
        Survey window.

    """
    _window = SurveyWindow(
        mask=lambda pos:\
            spherical_indicator(pos, boxsize/2) * sky_mask(pos, fsky)
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
    _xi_ell = window.correlation_function_multipoles(ORDERS)
    _pk_ell = window.power_multipoles

    return _xi_ell, _pk_ell


if __name__ == '__main__':

    PATHOUT = "./data/output/"

    params = parse_cli_args()
    pprint(params.__dict__)

    fsky = params.fsky
    nbar = params.nbar
    boxsize = params.boxsize
    padding = params.padding

    ORDERS = [0, 2, 4, 6, 8]

    window = synthesise()
    xi_ell, pk_ell = determine_window()

    np.save(
        f"{PATHOUT}mask_multipoles-{{:.2f}}sky-{{:.0f}}pad.npy"
        .format(fsky, padding),
        xi_ell
    )

    plt.close('all')

    plt.figure()

    for ell in ORDERS:
        plt.semilogx(
            pk_ell['k'], pk_ell[f'power_{ell}'],
            label=r'$\ell={}$'.format(ell)
        )

    plt.xlabel(r"$k$")
    plt.ylabel(r"$Q_\ell(k)$")
    plt.legend()

    plt.savefig(
        f"{PATHOUT}window-{{:.2f}}sky-{{:.0f}}pad.pdf".format(fsky, padding)
    )

    plt.figure()

    for ell in ORDERS:
        plt.semilogx(
            xi_ell['s'], xi_ell[f'correlation_{ell}'],
            label=r'$\ell={}$'.format(ell)
        )

    plt.xlabel(r"$s$")
    plt.ylabel(r"$Q_\ell(s)$")
    plt.legend()

    plt.savefig(
        f"{PATHOUT}mask-{{:.2f}}sky-{{:.0f}}pad.pdf".format(fsky, padding)
    )