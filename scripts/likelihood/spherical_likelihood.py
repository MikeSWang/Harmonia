"""Sample the spherical likelihood from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.cosmology import Cosmology

from likelihood_rc import PATHIN, PATHOUT, parse_external_args, script_name
from likelihood_rc import domain_cut
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import TwoPointFunction
from harmonia.reader import spherical_map_log_likelihood as sph_likelihood

# Cosmological input.
COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

# Survey specfications input.
SPECS_PATH = PATHIN/"specifications"
COUPLINGS_FILE = "couplings-(pivot={},kmax={},fsky={:.2f}).npy"

# Likelihood input.
FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"

# Survey catalogue input.
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]
TO_MESH_KWARGS = dict(resampler='tsc', compensated=True, interlaced=True)

# Global quantities.
simu_cosmo = None
external_couplings = None

fixed_params = None
sampled_params = None


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

    # Extract fixed parameter values and sampled parameter ranges.
    global fixed_params, sampled_params

    fixed_tag = ""
    fixed_params = {}
    with open(FIXED_PARAMS_FILE, 'r') as fixed_par_file:
        for name, values in eval(fixed_par_file.read()).items():
            fixed_tag += "{}={},".format(name, values[0])
            if name == 'fnl':
                name = 'non_gaussianity'
            fixed_params.update(
                {name : dict(zip(['spherical', 'cartesian'], values))}
            )
    ini_params.update({'fixed_params': fixed_tag.strip(",")})

    sampled_tag = ""
    sampled_params = {}
    with open(SAMPLED_PARAMS_FILE, 'r') as samp_par_file:
        for name, (ranges, num_sample) in eval(samp_par_file.read()).items():
            sampled_tag += "{}_prior=[{},{}],".format(name, *ranges)
            if name == 'fnl':
                name = 'non_gaussianity'
            sampled_params.update(
                {name: np.linspace(*ranges, num=num_sample+1)}
            )
    ini_params.update({'sampled_params': sampled_tag.strip(",")})

    rsd_tag = "rsd=on," if parsed_params.rsd else "rsd=off,"
    growth_rate = None if parsed_params.rsd else 0.
    ini_params.update({'growth_rate': growth_rate})

    ini_tag = "map={},kmax={},pivot={},{}{}{}".format(
        parsed_params.map, parsed_params.khyb, parsed_params.spherical_pivot,
        rsd_tag, sampled_tag, fixed_tag,
    ).strip(",")

    # Extract cosmology and survey specifications.
    global simu_cosmo, external_couplings

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    if parsed_params.load_couplings:
        external_couplings = np.load(
            SPECS_PATH/COUPLINGS_FILE.format(
                parsed_params.spherical_pivot,
                str(parsed_params.khyb).rstrip("0"),
                parsed_params.fsky
            )
        ).item()

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
    disc = DiscreteSpectrum(params['boxsize']/2, 'dirichlet', params['khyb'])

    two_point_model = TwoPointFunction(
        disc,
        redshift=params['redshift'],
        growth_rate=params['growth_rate'],
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
        overdensity_field = spherical_map.density_constrast()

        spherical_data = SphericalArray.build(
            disc=disc, filling=overdensity_field
        )

        # Construct spherical likelihood.
        sph_likelihood_kwargs = dict(
            two_point_model=two_point_model,
            spherical_data=spherical_data,
            mean_number_density=params['nbar'],
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

    filename = f"{params['input_catalogue']}-({filetag}).npy"

    np.save(PATHOUT/script_name/filename, results)


if __name__ == '__main__':

    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    finalise(output, tag)
