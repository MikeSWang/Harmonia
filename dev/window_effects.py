"""Window effects in measured power spectrum multipoles of simulated
catalogues.

"""
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product as iterprod
from pprint import pprint

import numpy as np
from nbodykit.lab import ConvolvedFFTPower, FKPCatalog

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import cartesian_to_spherical, confirm_directory_path
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)

PATHIN, PATHOUT = "./data/input/", "./data/output/"
SCRIPT_NAME = "window_effects"

INPUT_FILE_ROOT = "halos-(NG=0.,z=1.)-"
PAIR_NUMS = list(range(0, 11)) + list(range(12, 22))
PAIR_SUFFICES = ["L", "R"]
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


def parse_cli_args():

    cli_parser = ArgumentParser()

    cli_parser.add_argument('--fsky', type=float, default=1/3)
    cli_parser.add_argument('--split', action='store_true')
    cli_parser.add_argument('--nbar', type=float, default=2.4883e-4)
    cli_parser.add_argument('--contrast', type=float, default=40.)
    cli_parser.add_argument('--boxsize', type=float, default=1000.)
    cli_parser.add_argument('--mesh', type=int, default=256)

    return cli_parser.parse_args()


def sky_mask(cartesian_position, fraction, split=False):

    spherical_position = cartesian_to_spherical(cartesian_position)

    if split:
        mask = np.logical_or(
            np.logical_and(
                spherical_position[..., -1] <= fraction * (2*np.pi),
                spherical_position[:, 1] < np.pi/2
            ),
            np.logical_and(
                spherical_position[..., -1] >= (1 - fraction) * (2*np.pi),
                spherical_position[:, 1] >= np.pi/2
            )
        )
    else:
        mask = spherical_position[..., -1] <= fraction * (2*np.pi)

    return mask


def process():

    disc = DiscreteSpectrum(boxsize/2, 'dirichlet', KMAX)

    no_window_suite = defaultdict(list)
    windowed_suite = defaultdict(list)
    for num, suffix in iterprod(PAIR_NUMS, PAIR_SUFFICES):
        catalogue_path = "{}{}/{}{}{}.txt".format(
            PATHIN, "catalogues", INPUT_FILE_ROOT, num, suffix
        )

        data_catalogue = load_catalogue_from_file(
            catalogue_path, CATALOGUE_HEADINGS, boxsize
        )
        random_catalogue = RandomCatalogue(contrast*nbar, boxsize)

        data_catalogue['NZ'] = nbar * data_catalogue['Weight']
        random_catalogue['NZ'] = nbar * random_catalogue['Weight']

        # No window
        no_window_mesh = FKPCatalog(data_catalogue, random_catalogue)\
            .to_mesh(
                Nmesh=mesh, resampler='tsc', compensated=True, interlaced=True
            )
        no_window_power = ConvolvedFFTPower(no_window_mesh, ORDERS, kmax=KMAX).poles

        valid_bins = (
            ~np.isnan(no_window_power['modes']) \
            & ~np.equal(no_window_power['modes'], 0) \
            & ~(no_window_power['modes'] % 2)
        ).astype(bool)

        no_window_suite['k'].append([
            no_window_power['k'][valid_bins]
        ])
        no_window_suite['modes'].append([
            no_window_power['modes'][valid_bins]
        ])
        no_window_suite['shotnoise'].append([
            no_window_power.attrs['shotnoise']
        ])
        for ell in ORDERS:
            no_window_suite[f'power_{ell}'].append([
                no_window_power[f'power_{ell}'][valid_bins].real
            ])

        # With window
        data_catalogue['Weight'] *= sky_mask(
            data_catalogue['Position'] - boxsize/2, fsky, split=split
        )
        random_catalogue['Weight'] *= sky_mask(
            random_catalogue['Position'] - boxsize/2, fsky, split=split
        )

        spherical_map = SphericalMap(
            disc, data_catalogue,
            rand=random_catalogue,
            mean_density_data=nbar,
            mean_density_rand=contrast*nbar
        )

        windowed_mesh = spherical_map.pair.to_mesh(
            Nmesh=mesh, resampler='tsc', compensated=True, interlaced=True
        )
        windowed_power = ConvolvedFFTPower(windowed_mesh, ORDERS, kmax=KMAX).poles

        valid_bins = (
            ~np.isnan(windowed_power['modes']) \
            & ~np.equal(windowed_power['modes'], 0) \
            & ~np.equal(windowed_power['modes'], 1)
        ).astype(bool)

        windowed_suite['k'].append([
            windowed_power['k'][valid_bins]
        ])
        windowed_suite['modes'].append([
            windowed_power['modes'][valid_bins]
        ])
        windowed_suite['shotnoise'].append([
            windowed_power.attrs['shotnoise']
        ])
        for ell in ORDERS:
            windowed_suite[f'power_{ell}'].append([
                windowed_power[f'power_{ell}'][valid_bins].real
            ])

    return no_window_suite, windowed_suite


if __name__ == '__main__':

    REDSHIFT = 0.
    KMAX = 0.1
    ORDERS = [0, 2, 4]

    # Set script parameter.
    params = parse_cli_args()

    fsky = params.fsky
    split = params.split
    nbar = params.nbar
    contrast = params.contrast
    boxsize = params.boxsize
    mesh = params.mesh

    pprint(params.__dict__)
    script_tag = "fsky={:.2f}{},contrast={},mesh={}".format(
        fsky, split*"s", contrast, mesh
    )

    # Process catalogues.
    no_window_output, windowed_output = process()

    confirm_directory_path(PATHOUT + SCRIPT_NAME + "/")
    np.save(
        "".join([
            PATHOUT, SCRIPT_NAME, "/",
            "no_window_measurements", "-(", script_tag, f").npy"
        ]),
        no_window_output
    )
    np.save(
        "".join([
            PATHOUT, SCRIPT_NAME, "/",
            "windowed_measurements", "-(", script_tag, f").npy"
        ]),
        windowed_output
    )
