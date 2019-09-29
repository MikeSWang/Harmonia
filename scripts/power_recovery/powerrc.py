"""Power spectrum recovery runtime configuration.

This sets I/O paths and provides common parameters and functionalities to power
spectrum recovery scripts.

"""
import sys
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, "../../")

from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    get_filename,
)


def parse_cli_args(cli_parser):

    # Physical parameters
    cli_parser.add_argument('--nbar', type=float, default=1e-3)
    cli_parser.add_argument('--contrast', type=float, default=None)
    cli_parser.add_argument('--bias', type=float, default=2.)
    cli_parser.add_argument('--redshift', type=float, default=0.)
    cli_parser.add_argument('--zmax', type=float, default=0.05)
    cli_parser.add_argument('--rmax', type=float, default=150.)
    cli_parser.add_argument('--kmax', type=float, default=0.1)
    cli_parser.add_argument('--dk', type=float, default=1e-2)

    # Computing parameters
    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--expand', type=float, default=2.)
    cli_parser.add_argument('--meshgen', type=int, default=256)
    cli_parser.add_argument('--meshcal', type=int, default=256)

    # Program parameters
    cli_parser.add_argument('--niter', type=int, default=10)
    cli_parser.add_argument('--progid', default="")
    cli_parser.add_argument('--infile', default="halos-(NG=0.,z=1.)-0")

    return cli_parser.parse_args()


def aggregate(result):
    return {
        'Nk': np.sum(result['Nk'], axis=0),
        'k': np.average(result['k'], axis=0),
        'Pk': np.average(result['Pk'], axis=0),
        'Pshot': np.average(result['Pshot']),
        'ln': np.atleast_2d(result['ln'])[-1],
        'kln': np.atleast_2d(result['kln'])[-1],
        'Pln': np.average(result['Pln'], axis=0),
        'dk': np.std(result['k'], axis=0, ddof=1),
        'dPk': np.std(result['Pk'], axis=0, ddof=1),
        'dPln': np.std(result['Pln'], axis=0, ddof=1),
        'dof1': np.size(result['k'], axis=-1) - 1,
        'dof2': np.size(result['Pln'], axis=0) - 1,
    }


def quick_plot(output):

    dof1 = np.size(output['k'], axis=0) - 1
    dof2 = np.size(output['Pln'], axis=0) - 1

    results = {
        'Nk': np.sum(output['Nk'], axis=0),
        'k': np.average(output['k'], axis=0),
        'Pk': np.average(output['Pk'], axis=0),
        'Pshot': np.average(output['Pshot']),
        'ln': output['ln'],
        'kln': output['kln'],
        'Pln': np.average(output['Pln'], axis=0),
        'dk': np.std(output['k'], axis=0, ddof=1) / np.sqrt(dof1),
        'dPk': np.std(output['Pk'], axis=0, ddof=1) / np.sqrt(dof2),
        'dPln': np.std(output['Pln'], axis=0, ddof=1) / np.sqrt(dof2),
    }

    cartesian_result = plt.errorbar(
        results['k'],
        results['Pk'],
        xerr=results['dk'],
        yerr=results['dPk'],
        color='#0087BD',
        label='Cartesian',
    )

    spherical_result = plt.loglog(
        results['kln'],
        results['Pln'],
        color='#C40233',
        label='spherical',
    )

    for layer in [1, 2]:
        plt.fill_between(
            results['kln'],
            results['Pln'] - layer * results['dPln'],
            results['Pln'] + layer * results['dPln'],
            color=spherical_result[0].get_color(),
            alpha=(1/4)**layer,
        )

    for idx, dbl_indices in enumerate(results['ln']):
        if dbl_indices[0] == 0:
            plt.annotate(
                str(dbl_indices),
                xy=(results['kln'][idx], results['Pln'][idx]),
                verticalalignment='bottom',
                fontsize=6,
            )

    plt.axhline(
        y=results['Pshot'],
        linestyle='--',
        color=cartesian_result[0].get_color(),
        alpha=.5,
    )

    plt.xlim(left=0.99*results['kln'].min(), right=1.01*results['kln'].max())
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()


PATHIN = "./data/input/"
PATHOUT = "./data/output/"

filename = get_filename(sys.argv[0])

params = parse_cli_args(ArgumentParser())
