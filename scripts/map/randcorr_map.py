"""Generate random correlated hybrid maps.

"""
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nbodykit.lab import ConvolvedFFTPower, FKPCatalog, UniformCatalog

from map_rc import PATHOUT, script_name, domain_cut
from harmonia.algorithms import DiscreteSpectrum
from harmonia.algorithms import CartesianArray, SphericalArray
from harmonia.mapper import CartesianMap, SphericalMap
from harmonia.collections import (
    collate_data_files,
    confirm_directory_path,
    covar_to_corr,
    harmony,
    overwrite_protection,
    sort_dict_to_list,
)


def parse_args():
    """Parse command-line argument inputs.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameter space.

    """
    parser = ArgumentParser()

    parser.add_argument('--task', choices=['gen', 'agg'])
    parser.add_argument('--batchno', default='')

    parser.add_argument('--fsky', type=float, default=1.)

    parser.add_argument('--pivot', default='spectral')

    parser.add_argument('--kmin', type=float, default=None)
    parser.add_argument('--khyb', type=float, default=0.04)
    parser.add_argument('--kmax', type=float, default=0.1)

    parser.add_argument('--orders', type=int, nargs='+', default=[0])

    parser.add_argument('--nbar', type=float, default=2.5e-4)
    parser.add_argument('--contrast', type=float, default=10.)
    parser.add_argument('--boxsize', type=float, default=1000.)
    parser.add_argument('--dk', type=float, default=None)
    parser.add_argument('--mesh', type=int, default=256)
    parser.add_argument('--iter', type=int, default=100)

    return parser.parse_args()


def initialise():
    """Initialise program with parsed parameters.

    Returns
    -------
    str
        Initialisation tag.

    """
    param_dict = vars(params)

    pprint(param_dict)

    return (
        "fsky={:.2f},knots=[{},{},{}],orders={},boxsize={:.0f},iter={:d}"
    ).format(
        params.fsky, params.kmin, params.khyb, params.kmax,
        str(params.orders).replace(", ", ","), params.boxsize, params.iter
    )


def generate():
    """Generate FKP-paired random catalogues with the desired survey window
    and process them into power spectrum measurements.

    Returns
    -------
    results : dict
        Multipole measurements.

    """
    results = {'cmap': defaultdict(list), 'smap': defaultdict(list)}
    for _ in range(params.iter):
        data_catalogue = UniformCatalog(params.nbar, params.boxsize)
        rand_catalogue = UniformCatalog(
            params.contrast*params.nbar, params.boxsize
        )

        for catalogue in [data_catalogue, rand_catalogue]:
            catalogue['Selection'] *= domain_cut(
                catalogue['Position'] - [params.boxsize/2] * 3,
                params.boxsize/2, params.fsky,
            )
            catalogue['NZ'] = params.nbar * catalogue['Weight']

        catalogue_pair = FKPCatalog(data_catalogue, rand_catalogue)
        cartesian_map = CartesianMap(catalogue_pair, num_mesh=params.mesh)
        cartesian_power = cartesian_map.power_multipoles(
            params.orders, kmin=params.khyb, kmax=params.kmax, dk=params.dk
        )

        results['k'].append(cartesian_power['k'])
        results['Nk'].append(cartesian_power['Nk'])
        for ell in params.orders:
            results[f'power_{ell}'].append(cartesian_power[f'power_{ell}'])

    return results


if __name__ == '__main__':

    params = parse_args()
    tag = initialise()

