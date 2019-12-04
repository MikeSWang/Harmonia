"""Sample the spherical likelihood from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.cosmology import Cosmology

from likelihood_rc import PATHIN, PATHOUT, parsed_params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import TwoPointFunction
from harmonia.reader import spherical_map_log_likelihood as sph_likelihood

COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

COUPLINGS_FILE = PATHIN/"specifications"/""

FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"

CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]
TO_MESH_KWARGS = dict(resampler='tsc', compensated=True, interlaced=True)

simu_cosmo = None
external_couplings = None


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
    pprint(ini_params)

    # Extract fixed parameter values and sampled parameter ranges.
    global fixed_params, sampled_params

    fixed_tag = ""
    fixed_params = {}
    with open(FIXED_PARAMS_FILE, 'r') as fixed_par_file:
        for name, values in eval(fixed_par_file.read()).items():
            fixed_tag += "{}=[s{},c{}],".format(name, *values)
            fixed_params.update(
                {name : dict(zip(['spherical', 'cartesian'], values))}
            )

    sampled_tag = ""
    sampled_params = {}
    with open(SAMPLED_PARAMS_FILE, 'r') as samp_par_file:
        for name, (ranges, num_sample) in eval(samp_par_file.read()).items():
            sampled_tag += "{}_prior=[{},{}],".format(name, *ranges)
            sampled_params.update(
                {name: np.linspace(*ranges, num=num_sample+1)}
            )

    ini_tag = "map={},knots=[{},{}],pivots=[{},{}],{},{}".format(
        parsed_params.map, parsed_params.khyb, parsed_params.kmax,
        parsed_params.spherical_pivot, parsed_params.cartesian_pivot,
        sampled_tag, fixed_tag
    )

    # Extract cosmology and survey specifications.
    global simu_cosmo, external_couplings

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    external_couplings = None

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    disc = DiscreteSpectrum(params['boxsize']/2, 'dirichlet', params['khyb'])

    two_point_model = TwoPointFunction(
        disc,
        redshift=params['redshift'],
        cosmo=simu_cosmo,
        couplings=external_couplings
    )

    output_data = defaultdict(list)
    for file_suffix in ["L.txt", "R.txt"]:
        # Build map from loaded catalogue.
        catalogue_name = params['input_catalogue'] + file_suffix
        catalogue_path = PATHIN/"catalogues"/catalogue_name

        data_catalogue = load_catalogue_from_file(
            str(catalogue_path), CATALOGUE_HEADINGS, params['boxsize']
        )
        random_catalogue = RandomCatalogue(
            params['contrast']*params['nbar'], params['boxsize']
        )
        spherical_map = SphericalMap(
            disc, data_catalogue, rand=random_catalogue,
            mean_density_data=params['nbar'],
            mean_density_rand=params['contrast']*params['nbar']
        )

        # Compute measurements.
        overdensity_field = spherical_map.density_constrast()

        spherical_data = SphericalArray.build(
            disc=disc, filling=overdensity_field
        )

        # Construct spherical likelihood.
        sph_likelihood_kwargs = dict(
            two_point_model=two_point_model,
            spherical_data=spherical_data,
            nbar=params['nbar'],
            contrast=params['contrast'],
            pivot=params['spherical_pivot'],
            breakdown=params['breakdown'],
            independence=True,
        )
        for par_name, par_values in fixed_params.items():
            sph_likelihood_kwargs.update({par_name: par_values['spherical']})
        for par_name, par_values in sampled_params.items():
            sph_likelihood_kwargs.update({par_name: par_values})

        output_data['spherical_likelihood'].append(
            [sph_likelihood(**sph_likelihood_kwargs)]
        )

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

    filename = f"({params['input_catalogue']}{filetag}).npy"

    np.save(PATHOUT/script_name/filename, results)


if __name__ == '__main__':

    params, tag = initialise()
    output = process()
    finalise(output, tag)
