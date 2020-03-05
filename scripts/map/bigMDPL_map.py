"""Make maps from simulations.

"""
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pformat

import numpy as np
from mpi4py import MPI

from map_rc import PATHIN, PATHOUT, domain_cut, script_name
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    CartesianMap,
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)

CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


def parse_external_args():
    """Parse command line arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed parameters.

    """
    parser = ArgumentParser()

    parser.add_argument('--map', required=True)
    parser.add_argument('--input-catalogue', required=True)

    parser.add_argument('--fsky', type=float, default=1.)
    parser.add_argument('--kmin', type=float, default=None)
    parser.add_argument('--kmax', type=float, default=None)
    parser.add_argument('--rsd', action='store_true')
    parser.add_argument('--multipoles', type=int, nargs='+', default=[0])

    parser.add_argument('--nbar', type=float, default=9.287e-4)
    parser.add_argument('--contrast', type=float, default=10.)
    parser.add_argument('--boxsize', type=float, default=2500.)
    parser.add_argument('--dk', type=float, default=None)
    parser.add_argument('--mesh', type=int, default=256)

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

    ini_tag = "map={},fsky={:.2f},knots=[{},{}],rsd={}".format(
        parsed_params.map, parsed_params.fsky,
        parsed_params.kmin, parsed_params.kmax,
        parsed_params.rsd,
    ).strip(",")

    if comm is not None and comm.rank == 0:
        print("---Program parameters---", pformat(ini_params), sep="\n")

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    if params['map'].startswith('s'):
        disc = DiscreteSpectrum(
            params['boxsize']/2, 'dirichlet', params['kmax'], comm=comm
        )
    elif params['map'].startswith('c'):
        disc = DiscreteSpectrum(
            params['boxsize']/2, 'dirichlet', params['kmin'], comm=comm
        )

    # Build map from loaded catalogue.
    catalogue_name = params['input_catalogue'] + ".txt"
    catalogue_path = PATHIN/"catalogues"/catalogue_name

    data_catalogue = load_catalogue_from_file(
        str(catalogue_path), CATALOGUE_HEADINGS, params['boxsize'],
        add_vel=params['rsd']
    )
    random_catalogue = RandomCatalogue(
        params['contrast']*params['nbar'], params['boxsize']
    )

    for catalogue in [data_catalogue, random_catalogue]:
        catalogue['Selection'] *= domain_cut(
            catalogue['Position'], params['boxsize']/2, params['fsky']
        )
        catalogue['NZ'] = params['nbar'] * catalogue['Weight']

    spherical_map = SphericalMap(
        disc, data_catalogue, rand=random_catalogue,
        mean_density_data=params['nbar'],
        mean_density_rand=params['contrast']*params['nbar'],
        comm=comm
    )

    output_data = defaultdict(list)
    if params['map'].startswith('s'):
        output_data['smap'] = spherical_map.density_constrast()
    if params['map'].startswith('c'):
        cartesian_map = CartesianMap(
            spherical_map.pair, num_mesh=params['mesh']
        )
        if params['kmin']:
            kwargs = dict(
                orders=params['multipoles'],
                kmin=params['kmin'],
                kmax=params['kmax']
            )
        else:
            kwargs = dict(
                orders=params['multipoles'],
                kmax=params['kmax']
            )
        output_data['cmap'] = cartesian_map.power_multipoles(**kwargs)

    return output_data


def finalise(results, filetag):
    """Finalise program.

    Parameters
    ----------
    results : dict
        Output data.
    filetag : dict
        Output file tag.

    """
    assert confirm_directory_path(PATHOUT/script_name)

    filename = f"{params['input_catalogue']}-({filetag}).npy"

    np.save(PATHOUT/script_name/filename, results)


comm = None
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    finalise(output, tag)
