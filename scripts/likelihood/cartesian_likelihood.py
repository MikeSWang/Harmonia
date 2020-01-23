"""Sample the Cartesian likelihood from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.cosmology import Cosmology

from likelihood_rc import PATHIN, PATHOUT, DATAPATH, script_name
from likelihood_rc import domain_cut, parse_external_args
from harmonia.algorithms import CartesianArray, DiscreteSpectrum
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    CartesianMap,
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import WindowedCorrelation, WindowedPowerSpectrum
from harmonia.reader import cartesian_map_log_likelihood as cart_likelihood

# Cosmological input.
COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

# Survey specfications input.
SPECS_PATH = PATHIN/"specifications"

# Catalogue map input.
MAP_PATH = DATAPATH/"cartesian_map"

MASK_MULTIPOLES_FILE = "mask_multipoles-{:.2f}sky.npy"
WINDOW_MULTIPOLES_FILE = "window_multipoles-{:.2f}sky.npy"
FIDUCIAL_ESTIMATE_FILENAME = (
    "fiducial_estimate-(fsky={:.2f},knots=[{},{}],orders={}).npy"
)

# Likelihood input.
FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"

# Survey catalogue input.
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]

# Global quantities.
simu_cosmo = None
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
    ini_params.update({'rsd': rsd_tag.lstrip("rsd=").rstrip(",")})
    growth_rate = None if parsed_params.rsd else 0
    ini_params.update({'growth_rate': growth_rate})

    ini_tag = "map={},fsky={},knots=[{},{}],orders={},pivot={},{}{}{}{}"\
        .format(
            parsed_params.map, parsed_params.fsky,
            parsed_params.kmin, parsed_params.kmax,
            str(parsed_params.multipoles).replace(", ", ","), 
            parsed_params.cartesian_pivot, rsd_tag,
            sampled_tag, fixed_tag,
            bool(parsed_params.num_cov_est) 
                * f"ncov={parsed_params.num_cov_est}",
        ).strip(",")

    # Extract cosmology and survey specifications.
    global simu_cosmo, mask_multipoles, window_multipoles, window_correlator

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    mask_multipoles = np.load(
        SPECS_PATH/MASK_MULTIPOLES_FILE.format(parsed_params.fsky)
    ).item()

    window_multipoles = np.load(
        SPECS_PATH/WINDOW_MULTIPOLES_FILE.format(parsed_params.fsky)
    ).item()

    fiducial_estimate = np.load(
        SPECS_PATH/(
            FIDUCIAL_ESTIMATE_FILENAME.format(
                parsed_params.fsky,
                str(np.around(parsed_params.khyb, decimals=3)).rstrip("0"),
                str(np.around(parsed_params.kmax, decimals=3)).rstrip("0"),
                str(parsed_params.multipoles).replace(", ", ",")
            )
        )
    ).item()
    window_correlator = WindowedCorrelation(
        fiducial_estimate['fiducial_data']
    )
    window_correlator.windowed_correlation = \
        fiducial_estimate['fiducial_covariance']

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
        cosmo=simu_cosmo,
        mask_multipoles=mask_multipoles,
        window_multipoles=window_multipoles
    )

    map_file = params['input_catalogue'] \
        + "-(map={},knots=[{},{}],orders={},rsd={}).npy".format(
            params['map'], params['kmin'], params['kmax'], 
            str(params['multipoles']).replace(", ", ","), params['rsd']
        )
    map_data = np.load(MAP_PATH/map_file).item()

    output_data = defaultdict(list)
    for file_suffix in ["L.txt", "R.txt"]:
        # Load map data.
        cartesian_data = CartesianArray(
            filling=map_data[file_suffix],
            coord_key='k',
            variable_key_root='power_'
        )

        # Construct Cartesian likelihood.
        windowed_power_model.wavenumbers = cartesian_data.coord_array
        cart_likelihood_kwargs = dict(
            windowed_power_model=windowed_power_model,
            correlation_modeller=window_correlator,
            cartesian_data=cartesian_data,
            mean_number_density=params['nbar'],
            contrast=params['contrast'],
            pivot=params['cartesian_pivot'],
            orders=params['multipoles'],
            num_covar_sample=params['num_cov_est']
        )
        for par_name, par_values in fixed_params.items():
            cart_likelihood_kwargs.update({par_name: par_values['cartesian']})
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

    filename = f"{params['input_catalogue']}-({filetag}).npy"

    np.save(PATHOUT/script_name/filename, results)


if __name__ == '__main__':

    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    finalise(output, tag)
