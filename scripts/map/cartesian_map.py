"""Make Cartesian maps from simulations.

"""
from collections import defaultdict
from pprint import pformat

import numpy as np

from map_rc import PATHIN, PATHOUT, script_name
from map_rc import domain_cut, parse_external_args
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    CartesianMap,
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)

CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


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

    ini_tag = "map={},fsky={:.2f},knots=[{},{}],orders={},rsd={}".format(
        parsed_params.map, parsed_params.fsky,
        parsed_params.kmin, parsed_params.kmax,
        str(parsed_params.multipoles).replace(", ", ","),
        parsed_params.rsd
    ).strip(",")

    print("---Program parameters---", pformat(ini_params), "", sep="\n")

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    disc = DiscreteSpectrum(params['boxsize']/2, 'dirichlet', params['kmax'])

    output_data = defaultdict(list)
    file_suffices = [".txt"] if "halos" not in params['input_catalogue'] \
        else ["L.txt", "R.txt"]

    for file_suffix in file_suffices:
        # Build map from loaded catalogue.
        catalogue_name = params['input_catalogue'] + file_suffix
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
            mean_density_rand=params['contrast']*params['nbar']
        )
        cartesian_map = CartesianMap(
            spherical_map.pair, num_mesh=params['mesh']
        )

        # Compute measurements.
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

        output_data[file_suffix] = cartesian_map.power_multipoles(**kwargs)

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

    filename = f"{params['input_catalogue']}-({filetag},theta).npy"

    np.save(PATHOUT/script_name/filename, results)


if __name__ == '__main__':

    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    finalise(output, tag)
