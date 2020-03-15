"""Generate random correlated hybrid maps.

"""
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nbodykit.lab import UniformCatalog

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
    print("---Program parameters---", pformat(vars(params)), "", sep="\n")

    return (
        "fsky={:.2f},knots=[{},{},{}],orders={},boxsize={:.0f},iter={:d},theta"
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
    disc = DiscreteSpectrum(params.boxsize/2, 'dirichlet', params.khyb)

    results = {'smap': [], 'cmap': defaultdict(list)}
    for _ in range(params.iter):
        data_catalogue = UniformCatalog(params.nbar, params.boxsize)
        rand_catalogue = UniformCatalog(
            params.contrast*params.nbar, params.boxsize
        )

        for catalogue in [data_catalogue, rand_catalogue]:
            catalogue['Selection'] *= domain_cut(
                catalogue['Position'], params.boxsize/2, params.fsky,
            )
            catalogue['NZ'] = params.nbar * catalogue['Weight']

        spherical_map = SphericalMap(
            disc, data_catalogue, rand=rand_catalogue,
            mean_density_data=params.nbar,
            mean_density_rand=params.contrast*params.nbar
        )
        spherical_data = spherical_map.density_constrast()

        results['smap'].append(spherical_data)

        cartesian_map = CartesianMap(spherical_map.pair, num_mesh=params.mesh)
        cartesian_power = cartesian_map.power_multipoles(
            params.orders, kmin=params.khyb, kmax=params.kmax, dk=params.dk
        )

        for key, value in cartesian_power.item():
            results['cmap'][key].append(value)

    return results


if __name__ == '__main__':

    confirm_directory_path(PATHOUT/script_name)

    params = parse_args()
    tag = initialise()

    if params.task == 'gen':

        output = generate()
        np.save(
            PATHOUT/script_name/"rancorr-({}){}.npy"
            .format(
                tag, '-[{}]'.format(params.batchno) * bool(params.batchno)
            ),
            output
        )
