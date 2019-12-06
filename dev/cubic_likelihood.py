"""Sample the cubic likelihood from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.cosmology import Cosmology
from nbodykit.lab import FKPCatalog

from likelihood_rc import PATHIN, PATHOUT, parsed_params, script_name
from harmonia.algorithms import CartesianArray
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    CartesianMap,
    RandomCatalogue,
    load_catalogue_from_file,
)
from harmonia.reader import WindowedCorrelation, WindowedPowerSpectrum
from harmonia.reader import cartesian_map_log_likelihood as cart_likelihood

# Cosmological input.
COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

# Likelihood input.
FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"

# Survey catalogue input.
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]

# Global quantities.
ORDERS = [0]

simu_cosmo = None

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
            fixed_tag += "{}={},".format(name, values[-1])
            if name == 'fnl':
                name = 'non_gaussianity'
            fixed_params.update(
                {name : dict(zip([None, 'cubic'], values))}
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

    ini_tag = "map={},kmax={},{},{}".format(
        parsed_params.map, parsed_params.kmax, sampled_tag, fixed_tag
    )

    # Extract cosmology and survey specifications.
    global simu_cosmo

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

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
    windowed_power_model = WindowedPowerSpectrum(
        redshift=params['redshift'],
        growth_rate=params['growth_rate'],
        cosmo=simu_cosmo
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
        data_catalogue['NZ'] = params['nbar'] * data_catalogue['Weight']
        random_catalogue['NZ'] = params['nbar'] * random_catalogue['Weight']

        catalogue_pair = FKPCatalog(data_catalogue, random_catalogue)
        cartesian_map = CartesianMap(catalogue_pair, num_mesh=params['mesh'])

        # Cartesian measurements.
        cartesian_multipoles = cartesian_map.power_multipoles(
            orders=ORDERS, kmax=params['kmax']
        )

        cartesian_data = CartesianArray(
            filling=cartesian_multipoles,
            coord_key='k',
            var_key_root='power_'
        )

        # Construct Cartesian likelihood.
        windowed_power_model.wavenumbers = cartesian_data.coord_array
        window_correlator = WindowedCorrelation(
            {
                'k': cartesian_data.coord_array,
                'power_0': np.ones_like(cartesian_data.coord_array),
            }
        )
        window_correlator.windowed_correlation = np.diag(
            2/cartesian_multipoles['Nk']
        )

        cart_likelihood_kwargs = dict(
            windowed_power_model=windowed_power_model,
            correlation_modeller=window_correlator,
            cartesian_data=cartesian_data,
            mean_number_density=params['nbar'],
            contrast=params['contrast'],
            pivot=params['cartesian_pivot'],
            orders=params['multipoles'],
        )
        for par_name, par_values in fixed_params.items():
            cart_likelihood_kwargs.update({par_name: par_values['cubic']})
        for par_name, par_values in sampled_params.items():
            cart_likelihood_kwargs.update({par_name: par_values})

        output_data['cartesian_likelihood'].append(
            [cart_likelihood(**cart_likelihood_kwargs)]
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
