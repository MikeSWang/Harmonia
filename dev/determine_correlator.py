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

from harmonia.mapper import NBKCatalogue, load_catalogue_from_file
from harmonia.collections import cartesian_to_spherical
from harmonia.collections import collate_data_files, confirm_directory_path
from harmonia.collections import harmony

plt.style.use(harmony)
sns.set(style='ticks', font='serif')

PATHIN = Path("./data/input/")
PATHOUT = Path("./data/output/")
SCRIPT_NAME = "window_correlator"

CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]

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
    parser.add_argument('--input-catalogue', default='')
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

    if params.input_catalogue:
        for file_suffix in ["L.txt", "R.txt"]:
            catalogue_name = params.input_catalogue + file_suffix
            catalogue_path = PATHIN/"catalogues"/catalogue_name

            data_catalogue = load_catalogue_from_file(
                str(catalogue_path), CATALOGUE_HEADINGS, params['boxsize']
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

    else:
        for run in range(niter):
            if params.rand_samp:
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
    """Export aggregated program results.

    Returns
    -------
    collated_output : dict
        Aggregated output data.
    filename : int
        Aggregated data file name.

    """
    collated_output, file_count, _ = collate_data_files(
        f"{str(PATHOUT/SCRIPT_NAME)}/{FILE_ROOT}*{tag}*.npy"
        .replace("=[", "=[[]").replace("],", "[]],"),
        'npy'
    )

    invalid_bins = [Nk_array[0] % 2 for Nk_array in collated_output['Nk']]
    for var, vals in collated_output.items():
        collated_output[var] = [
            val[start:] for val, start in zip(vals, invalid_bins)
        ]

    filename = "{}-({})".format(
        FILE_ROOT, tag.replace(f"iter={niter}", f"iter={niter*file_count}")
    )
    np.save(PATHOUT/SCRIPT_NAME/"collated"/f"{filename}.npy", collated_output)

    return collated_output, filename


def extract(results):
    """Extract fiducial covariance estimate.

    Paramaters
    ----------
    results : dict
        Aggregated results.

    Returns
    -------
    estimate : dict
        Extracted covariance estimate.
    corr : dict
        Extracted sample correlation.

    """
    mean_data = {'k': np.mean(results['k'], axis=0)}
    mean_data.update(
        {
            f'power_{ell}': np.mean(results[f'power_{ell}'], axis=0)
            for ell in orders
        }
    )

    covar = np.cov(
        np.hstack((results[f'power_{ell}'] for ell in orders)),
        rowvar=False, ddof=1
    )

    corr = np.corrcoef(
        np.hstack((results[f'power_{ell}'] for ell in orders)), rowvar=False
    )

    error = {
        f'dpower_{ell}': np.std(results[f'power_{ell}'], axis=0, ddof=1)
            / np.sqrt(np.size(results[f'power_{ell}'], axis=0))
        for ell in orders
    }

    estimate = {
        'fiducial_data': mean_data,
        'fiducial_covariance': covar,
        'error': error
    }

    return estimate, corr


params = parse_args()

if params.input_catalogue:
    FILE_ROOT = params.input_catalogue
elif params.rand_samp:
    FILE_ROOT = "correlated_rsamples"
else:
    FILE_ROOT = "correlated_csamples"

    matter_power_spectrum = cosmology.LinearPower(
        cosmology.Planck15, redshift=0.
    )

if params.task.startswith('gen'):

    tag = initialise()
    output = process()

    confirm_directory_path(PATHOUT/SCRIPT_NAME)
    if params.sessionid:
        np.save(
            PATHOUT/SCRIPT_NAME/"{}-({})-[{}].npy"
            .format(FILE_ROOT, tag, params.sessionid),
            output
        )
    else:
        np.save(
            PATHOUT/SCRIPT_NAME/"{}-({}).npy".format(FILE_ROOT, tag), output
        )

elif params.task.startswith('agg'):
    confirm_directory_path(PATHOUT/SCRIPT_NAME/"collated")
    confirm_directory_path(PATHOUT/SCRIPT_NAME/"extracted")

    tag = initialise()
    output, name = export()

    fiducial_estimate, fiducial_corr = extract(output)

    np.save(
        PATHOUT/SCRIPT_NAME/"extracted"/(
            f"{name}.npy".replace(FILE_ROOT, "fiducial_estimate")
        ),
        fiducial_estimate
    )

    sns.heatmap(fiducial_corr, square=True, cmap='YlGn')
    plt.savefig(PATHOUT/SCRIPT_NAME/f"{name}.pdf")

else:
    tag = initialise()

    count = 32
    name = "{}-({})".format(
        FILE_ROOT, tag.replace(f"iter={niter}", f"iter={niter*count}")
    )
    output = np.load(PATHOUT/SCRIPT_NAME/"collated"/f"{name}.npy").item()

    fiducial_estimate, fiducial_corr = extract(output)

    np.save(
        PATHOUT/SCRIPT_NAME/"extracted"/(
            f"{name}.npy".replace(FILE_ROOT, "fiducial_estimate")
        ),
        fiducial_estimate
    )

    sns.heatmap(fiducial_corr, square=True, cmap='YlGn')
    plt.savefig(PATHOUT/SCRIPT_NAME/f"{name}.pdf")
