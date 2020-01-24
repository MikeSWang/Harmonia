"""Make spherical maps from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np

from map_rc import PATHIN, PATHOUT, script_name
from map_rc import domain_cut, parse_external_args
from harmonia.algorithms import DiscreteSpectrum
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)

CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]
TO_MESH_KWARGS = dict(resampler='tsc', compensated=True, interlaced=True)


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

    ini_tag = "map={},fsky={},knots=[{},{}],rsd={}".format(
        parsed_params.map, parsed_params.fsky,
        parsed_params.kmin, parsed_params.kmax,
        parsed_params.rsd,
    ).strip(",")

    pprint(ini_params)
    print("\n")

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
    for file_suffix in ["L.txt", "R.txt"]:
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

        # Compute measurements.
        output_data[file_suffix] = spherical_map.density_constrast()

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


if __name__ == '__main__':

    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    finalise(output, tag)
