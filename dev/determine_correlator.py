"""Determine the correlation induced by the survey window.

"""
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nbodykit.lab import cosmology
from nbodykit.lab import ConvolvedFFTPower, FKPCatalog, UniformCatalog

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

from harmonia.mapper import NBKCatalogue
from harmonia.collections import cartesian_to_spherical
from harmonia.collections import collate_data_files, confirm_directory_path
from harmonia.collections import harmony

plt.style.use(harmony)
sns.set(style='ticks', font='serif')

PATHOUT = Path("./data/output/")
SCRIPT_NAME = "window_correlator"

rand_samp = None
fsky, split = None, None
khyb, kmax, orders = None, None, None
nbar, contrast = None, None
boxsize, nmesh, niter = None, None, None


def domain_cut(cartesian_position, radius, fraction, split=False):
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
    spherical_position = cartesian_to_spherical(cartesian_position)

    veto = spherical_position[:, 0] <= radius
    if split:
        veto *= np.logical_or(
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
        veto *= spherical_position[:, 2] <= fraction * (2*np.pi)

    return veto


def parse_args():
    """Parse command-line argument inputs.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameter space.

    """
    parser = ArgumentParser()

    parser.add_argument('--task', required=True)
    parser.add_argument('--rand-samp', action='store_true')
    parser.add_argument('--sessionid', default='')

    parser.add_argument('--fsky', type=float, default=1.)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--khyb', type=float, default=0.05)
    parser.add_argument('--kmax', type=float, default=0.1)
    parser.add_argument('--orders', type=int, nargs='+', default=[0, 2, 4])
    parser.add_argument('--nbar', type=float, default=1e-3)
    parser.add_argument('--contrast', type=float, default=10.)
    parser.add_argument('--boxsize', type=float, default=1000.)
    parser.add_argument('--nmesh', type=int, default=256)
    parser.add_argument('--niter', type=int, default=50)

    return parser.parse_args()


def initialise():
    """Initialise program with parsed parameters.

    Returns
    -------
    str
        Initialisation tag.

    """
    param_dict = vars(params)
    for par_name, par_val in param_dict.items():
        globals()[par_name] = par_val

    pprint(param_dict)
    print("\n")

    return "fsky={0}{4},knots=[{1},{2}],boxsize={3},mesh={5},iter={6}".format(
        np.around(fsky, decimals=2),
        np.around(khyb, decimals=2),
        np.around(kmax, decimals=2),
        np.int(boxsize), split*"s", nmesh, niter
    )


def process():
    """Synthesise a FKP catalogue pair with the desired survey window and
    process them into power spectrum measurements.

    Returns
    -------
    results : dict
        Multipole measurements.

    """
    results = defaultdict(list)
    for run in range(niter):
        if rand_samp:
            data_catalogue = UniformCatalog(nbar, boxsize)
        else:
            data_catalogue = NBKCatalogue(
                matter_power_spectrum, nbar, boxsize, nmesh
            )
        rand_catalogue = UniformCatalog(contrast*nbar, boxsize)

        for catalogue in [data_catalogue, rand_catalogue]:
            catalogue['Selection'] *= domain_cut(
                catalogue['Position'] - [boxsize/2] * 3,
                boxsize/2, fsky, split=split
            )
            catalogue['NZ'] = nbar * catalogue['Weight']

        catalogue_pair = FKPCatalog(data_catalogue, rand_catalogue)
        catalogue_mesh = catalogue_pair.to_mesh(
            Nmesh=nmesh, resampler='tsc', compensated=True, interlaced=True
        )

        multipoles = ConvolvedFFTPower(
            catalogue_mesh, poles=orders, kmin=khyb, kmax=kmax
        ).poles
        valid_bins = (
            ~np.equal(multipoles['modes'], 0)
            & ~np.equal(multipoles['modes'], 1)
        )

        results['k'].append(multipoles['k'][valid_bins])
        results['Nk'].append(multipoles['modes'][valid_bins])
        for ell in orders:
            results[f'power_{ell}'].append(
                multipoles[f'power_{ell}'][valid_bins].real
            )

    return results


def export():

    collated_output, _, _ = collate_data_files(
        f"{str(PATHOUT/SCRIPT_NAME)}/{FILE_ROOT}*{tag}*.npy"
        .replace("=[", "=[[]").replace("],", "[]],"),
        'npy'
    )

    invalid_bins = [Nk_array[0] % 2 for Nk_array in collated_output['Nk']]
    for var, vals in collated_output.items():
        collated_output[var] = [
            val[start:] for val, start in zip(vals, invalid_bins)
        ]

    return collated_output


params = parse_args()

FILE_ROOT = "correlated_rsamples" if rand_samp else "correlated_csamples"

if params.task.startswith('gen'):

    matter_power_spectrum = cosmology.LinearPower(
        cosmology.Planck15, redshift=0.
    )

    tag = initialise()
    output = process()

    confirm_directory_path(PATHOUT/SCRIPT_NAME)
    np.save(
        PATHOUT/SCRIPT_NAME/"{}-({})-[{}].npy"
        .format(FILE_ROOT, tag, params.sessionid),
        output
    )
elif params.task.startswith('agg'):

    tag = initialise()
    output = export()
