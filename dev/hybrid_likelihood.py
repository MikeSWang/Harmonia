"""Sample the hybrid likelihoods from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.cosmology import Cosmology

from likelihood_rc import PATHIN, PATHOUT, parsed_params, script_name
from harmonia.algorithms import (
    CartesianArray,
    DiscreteSpectrum,
    SphericalArray,
)
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    CartesianMap,
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import (
    TwoPointFunction,
    WindowedCorrelation,
    WindowedPowerSpectrum,
)
from harmonia.reader import cartesian_map_log_likelihood as cart_likelihood
from harmonia.reader import spherical_map_log_likelihood as sph_likelihood

# Cosmological input.
COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

# Survey specfications input.
SPECS_PATH = PATHIN/"specifications"

COUPLINGS_FILE = SPECS_PATH/""
MASK_MULTIPOLES_FILE = SPECS_PATH/"mask_multipoles-1.00sky.npy"
WINDOW_MULTIPOLES_FILE = SPECS_PATH/"window_multipoles-1.00sky.npy"
FIDUCIAL_ESTIMATE_FILE = SPECS_PATH/"fiducial_estimation-1.00sky.npy"

# Likelihood input.
FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"

# Survey catalogue input.
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]

# Global quantities.
simu_cosmo = None
external_couplings = None
mask_multipoles = None
window_multipoles = None
window_correlator = None

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

    ini_tag = "map={},pivots=[{},{}],knots=[{},{}],{},{}".format(
        parsed_params.map,
        parsed_params.spherical_pivot, parsed_params.cartesian_pivot,
        parsed_params.khyb, parsed_params.kmax,
        sampled_tag, fixed_tag
    )

    # Extract cosmology and survey specifications.
    global simu_cosmo, external_couplings, mask_multipoles, \
        window_multipoles, window_correlator

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    external_couplings = np.load(COUPLINGS_FILE).item()

    mask_multipoles = np.load(MASK_MULTIPOLES_FILE).item()
    window_multipoles = np.load(WINDOW_MULTIPOLES_FILE).item()

    fiducial_estimate = np.load(FIDUCIAL_ESTIMATE_FILE).item()
    window_correlator = WindowedCorrelation(
        fiducial_estimate['fiducial_data']
    )
    window_correlator.windowed_correlation = \
        fiducial_estimate['fiducial_covariance']

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    disc = DiscreteSpectrum(params['boxsize']/2, 'dirichlet', params['khyb'])

    windowed_power_model = WindowedPowerSpectrum(
        redshift=params['redshift'],
        growth_rate=0.,
        cosmo=simu_cosmo,
        mask_multipoles=mask_multipoles,
        window_multipoles=window_multipoles
    )
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
        cartesian_map = CartesianMap(
            spherical_map.pair, num_mesh=params['mesh']
        )

        # Spherical measurements.
        overdensity_field = spherical_map.density_constrast()

        spherical_data = SphericalArray.build(
            disc=disc, filling=overdensity_field
        )

        # Cartesian measurements.
        cartesian_multipoles = cartesian_map.power_multipoles(
            orders=params['multipoles'],
            kmin=params['khyb'],
            kmax=params['kmax']
        )

        cartesian_data = CartesianArray(
            filling=cartesian_multipoles,
            coord_key='k',
            var_key_root='power_'
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

        # Construct Cartesian likelihood.
        windowed_power_model.wavenumbers = cartesian_data.coord_array
        cart_likelihood_kwargs = dict(
            windowed_power_model=windowed_power_model,
            correlation_modeller=window_correlator,
            cartesian_data=cartesian_data,
            nbar=params['nbar'],
            contrast=params['contrast'],
            pivot=params['cartesian_pivot'],
            orders=params['multipoles'],
        )
        for par_name, par_values in fixed_params.items():
            cart_likelihood_kwargs.update({par_name: par_values['cartesian']})
        for par_name, par_values in sampled_params.items():
            cart_likelihood_kwargs.update({par_name: par_values})

        output_data['cartesian_likelihood'].append(
            [cart_likelihood(**cart_likelihood_kwargs)]
        )

        output_data['data_vector'].append(
            np.concatenate(
                (
                    spherical_data.unfold(
                        params['spherical_pivot'], return_only='data'
                    ),
                    cartesian_data.unfold(
                        params['cartesian_pivot'], return_only='data'
                    ),
                )
            )
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
