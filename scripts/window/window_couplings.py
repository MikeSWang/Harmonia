"""Predict 2-point function values from baseline cosmological parameters
and survey specifications.

"""
import logging
import sys
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from mpi4py import MPI
from nbodykit.cosmology import Cosmology

from window_rc import PATHIN, PATHOUT, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import confirm_directory_path
from harmonia.reader import TwoPointFunction

COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"


def angular_mask(fraction, split_caps=False):
    """Define the angular mask.

    Parameters
    ----------
    fraction : float
        Sky fraction.
    split : bool, optional
        If `True` (default is `False`), the mask is split between north and
        south polar caps in antipodal regions.

    Returns
    -------
    callable
        Angular mask as a function of the angular coordinates.

    """
    def _mask(theta, phi):

        if split_caps:
            mask_val = np.logical_or(
                np.logical_and(
                    phi <= fraction * (2*np.pi), theta < np.pi/2
                ),
                np.logical_and(
                    phi >= (1 - fraction) * (2*np.pi), theta >= np.pi/2
                )
            )
        else:
            mask_val = (phi <= fraction * (2*np.pi))

        return mask_val

    return _mask


def parse_args():
    """Parse command-line argument inputs.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameter space.

    """
    parser = ArgumentParser()

    parser.add_argument('--fsky', type=float, default=1.)
    parser.add_argument('--rsd', action='store_true')
    parser.add_argument('--redshift', type=float, default=1.)
    parser.add_argument('--kmax', type=float, default=0.1)
    parser.add_argument('--rmax', type=float, default=500.)

    return parser.parse_args()


def initialise():
    """Initialise program by setting parameters.

    Returns
    -------
    ini_params : dict
        Initialised parameters.
    ini_tag :
        Initialised program tag.

    """
    ini_params = parsed_params.__dict__

    growth_rate = None if parsed_params.rsd else 0
    ini_params.update({'growth_rate': growth_rate})

    ini_tag = "fsky={:.2f},rmax={}.,kmax={},rsd={}".format(
        parsed_params.fsky, int(np.around(parsed_params.rmax)),
        parsed_params.kmax, parsed_params.rsd
    )

    return ini_params, ini_tag


def process():
    """Program process.

    Returns
    -------
    output_data : dict
        Program output.

    """
    logging_handler = logging.StreamHandler(sys.stdout)
    logging_formatter = logging.Formatter(
        fmt='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging_handler.setFormatter(logging_formatter)

    logger = logging.getLogger("TwoPointFunction")
    logger.addHandler(logging_handler)
    logger.setLevel(logging.INFO)

    if COMM is not None and COMM.rank == 0:
        pprint(params)
        print(f"\nRunning on {COMM.size} process(es).\n")

    disc = DiscreteSpectrum(params['rmax'], 'Dirichlet', params['kmax'])

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    survey_specs = {
        'mask': angular_mask(params['fsky']),
        'selection': None,
        'weight': None,
        'weight_derivative': None
    }

    kwargs = dict(
        redshift=params['redshift'],
        growth_rate=params['growth_rate'],
        cosmo=cosmo,
        survey_specs=survey_specs,
        comm=COMM,
    )

    two_points = TwoPointFunction(disc, **kwargs)

    output_data = two_points.couplings

    return output_data


def finalise():
    """Finalise program.

    """
    assert confirm_directory_path(PATHOUT/script_name)

    filename = f"couplings-({tag}).npy"
    if COMM is None or COMM.rank == 0:
        np.save(PATHOUT/script_name/filename, output)
    else:
        pass


if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    parsed_params = parse_args()
    params, tag = initialise()
    output = process()
    finalise()
