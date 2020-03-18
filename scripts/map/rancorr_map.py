"""Generate random correlated hybrid maps.

"""
import logging
import time
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nbodykit.lab import UniformCatalog
from tqdm import tqdm

from map_rc import PATHOUT, script_name
from map_rc import domain_cut
from harmonia.algorithms import DiscreteSpectrum
from harmonia.algorithms import CartesianArray, SphericalArray
from harmonia.mapper import CartesianMap, SphericalMap
from harmonia.collections import (
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

    global disc

    disc = DiscreteSpectrum(params.boxsize/2, 'dirichlet', params.khyb)

    time.sleep(1)

    return (
        "fsky={:.2f},knots=[{},{},{}],orders={},boxsize={:.0f},iter={:d},theta"
    ).format(
        params.fsky, params.kmin, params.khyb, params.kmax,
        str(params.orders).replace(", ", ","), params.boxsize, params.iter
    )


def generate():
    """Generate FKP-paired random catalogues with the desired survey window
    and process them into clustering map measurements.

    Returns
    -------
    results : dict
        Multipole measurements.

    """
    logging.getLogger().setLevel(logging.WARNING)

    def _generate(results={'smap': [], 'cmap': defaultdict(list)}):

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

        for key, value in cartesian_power.items():
            results['cmap'][key].append(value)

        return results

    for _ in tqdm(range(params.iter)):
        output = _generate()

    np.save(
        PATHOUT/script_name/(
            "rancorr-({}){}.npy".format(
                tag, '-[{}]'.format(params.batchno) * bool(params.batchno)
            )
        ),
        output
    )

    return output


def extract():
    """Generate FKP-paired random catalogues with the desired survey window
    and process them into power spectrum measurements.

    """
    fileroot = str(PATHOUT/script_name/f"rancorr-({tag})")
    searchroot = fileroot.replace("=[", "=[[]").replace("],", "[]],")

    hybrid_data, data_count = [], 0
    for datafile in glob(searchroot + "*.npy"):
        data = np.load(datafile).item()
        for idx_in_batch in range(params.iter):
            smap = SphericalArray.build(
                filling=sort_dict_to_list(data['smap'][idx_in_batch]),
                disc=disc
            )
            sdata = smap.unfold(params.pivot, return_only='data')

            cmap = CartesianArray(
                filling={k: v[idx_in_batch] for k, v in data['cmap'].items()},
                coord_key='k',
                variable_key_root='power_'
            )
            cdata = cmap.unfold('coord', return_only='data')

            hybrid_data.append(np.hstack((sdata, cdata)))
            data_count += 1

    filedir = PATHOUT/script_name/"collated"
    filename = f"rancorr-({tag}).npy".replace(
        f"iter={params.iter}", f"iter={data_count}"
    )
    if overwrite_protection(filedir, filename):
        np.save(filedir/filename, hybrid_data)

    mean_data = np.average(hybrid_data, axis=0)
    sample_cov = np.conj(hybrid_data - mean_data).T \
        @ np.asarray(hybrid_data - mean_data) \
        / len(hybrid_data)
    sample_corr = covar_to_corr(sample_cov)

    return sample_cov, sample_corr


def visualise(thredshold=0., zoom=None):
    """Visualise empirical covariance--correlation matrix.

    """
    plt.close('all')
    plt.style.use(harmony)
    sns.set(style='ticks', font='serif')

    reflection_mask = np.triu_indices(len(hybrid_covar), k=1)
    view_corr = hybrid_corr.real
    view_corr[reflection_mask] = hybrid_corr.imag[reflection_mask]

    view_offcorr = view_corr[zoom:, :zoom]  # real part only

    view_corr[np.abs(view_corr) < thredshold] = 0.
    sns.heatmap(
        view_corr, square=True, rasterized=True, cmap='coolwarm'
    )

    if zoom:
        plt.figure()
        sns.heatmap(view_offcorr, cmap='coolwarm', vmin=-0.1, vmax=0.1)

    return


if __name__ == '__main__':

    params = parse_args()
    tag = initialise()

    confirm_directory_path(PATHOUT/script_name/"collated")
    if params.task == 'gen':
        output = generate()
    if params.task == 'agg':
        hybrid_covar, hybrid_corr = extract()
        visualise(zoom=456)
