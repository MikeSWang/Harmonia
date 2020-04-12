"""Sample the hybrid likelihoods from simulations.

"""
from collections import defaultdict
from pprint import pformat

import numpy as np
from mpi4py import MPI
from nbodykit.cosmology import Cosmology

from likelihood_rc import PATHIN, PATHOUT, DATAPATH, logger, script_name
from likelihood_rc import parse_external_args
from harmonia.algorithms import (
    CartesianArray,
    DiscreteSpectrum,
    SphericalArray,
)
from harmonia.collections import confirm_directory_path
from harmonia.reader import (
    TwoPointFunction,
    WindowedCorrelation,
    WindowedPowerSpectrum,
)
from harmonia.reader import cartesian_map_log_likelihood as cart_likelihood
from harmonia.reader import spherical_map_log_likelihood as sph_likelihood

# Cosmological input.
COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"

# Catalogue map input.
SMAP_PATH = DATAPATH/"spherical_map"
CMAP_PATH = DATAPATH/"cartesian_map"

# Survey specfications input.
SPECS_PATH = PATHIN/"specifications"

COUPLINGS_FILE = "couplings-(fsky={:.2f},kmax={},theta).npy"
MASK_MULTIPOLES_FILE = "mask_multipoles-{:.2f}sky-theta.npy"
WINDOW_MULTIPOLES_FILE = "window_multipoles-{:.2f}sky-theta.npy"
FIDUCIAL_ESTIMATE_FILENAME = (
    "fiducial_estimate-(fsky={:.2f},knots=[{},{}],orders={},theta).npy"
)

# Likelihood input.
FIXED_PARAMS_FILE = PATHIN/"fixed_parameters.txt"
SAMPLED_PARAMS_FILE = PATHIN/"sampled_parameters.txt"
COMPRESSION_FILE = PATHIN/"rotation.npy"

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

    exclude_monopole = (0,) * parsed_params.nomono
    ini_params.update({'exclude_degrees': exclude_monopole})

    # Extract fixed parameter values and sampled parameter ranges.
    global fixed_params, sampled_params

    fixed_tag = ""
    fixed_params = {}
    with open(FIXED_PARAMS_FILE, 'r') as fixed_par_file:
        for name, values in eval(fixed_par_file.read()).items():
            fixed_tag += "{}=[s{},c{}],".format(name, *values)
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

    growth_rate = None if parsed_params.rsd else 0
    ini_params.update({'growth_rate': growth_rate})

    est_tag = bool(parsed_params.num_cov_est) \
        * f"ncov={parsed_params.num_cov_est},"

    ini_tag = (
        "map={},fsky={:.2f},knots=[{},{}],rsd={},orders={},{}{}{}{},theta"
    ).format(
        parsed_params.map, parsed_params.fsky,
        parsed_params.khyb, parsed_params.kmax,
        parsed_params.rsd,
        str(parsed_params.multipoles).replace(", ", ","),
        sampled_tag, fixed_tag, est_tag, parsed_params.nomono * "nomono"
    ).strip(",")

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

    ini_params.update({'sph_mode_independence': True})
    if parsed_params.load_couplings:
        try:
            external_couplings = np.load(
                SPECS_PATH/COUPLINGS_FILE.format(
                    parsed_params.fsky, str(parsed_params.khyb).rstrip("0"),
                )
            ).item()
        except FileNotFoundError:
            pass
        else:
            ini_params.update({'sph_mode_independence': False})

    mask_multipoles = np.load(
        SPECS_PATH/MASK_MULTIPOLES_FILE.format(parsed_params.fsky)
    ).item()

    window_multipoles = np.load(
        SPECS_PATH/WINDOW_MULTIPOLES_FILE.format(parsed_params.fsky)
    ).item()

    fiducial_estimate = np.load(
        SPECS_PATH/(
            FIDUCIAL_ESTIMATE_FILENAME.format(
                parsed_params.fsky, parsed_params.khyb, parsed_params.kmax,
                str(parsed_params.multipoles).replace(", ", ",")
            )
        )
    ).item()
    window_correlator = WindowedCorrelation(
        fiducial_estimate['fiducial_data']
    )
    window_correlator.windowed_correlation = \
        fiducial_estimate['fiducial_covariance']

    if comm.rank == 0:
        print("---Program parameters---", pformat(ini_params), "", sep="\n")

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    disc = DiscreteSpectrum(
        params['boxsize']/2, 'dirichlet', params['khyb'], comm=comm
    )

    windowed_power_model = WindowedPowerSpectrum(
        redshift=params['redshift'],
        growth_rate=params['growth_rate'],
        cosmo=simu_cosmo,
        mask_multipoles=mask_multipoles,
        window_multipoles=window_multipoles
    )
    two_point_model = TwoPointFunction(
        disc,
        redshift=params['redshift'],
        growth_rate=params['growth_rate'],
        cosmo=simu_cosmo,
        couplings=external_couplings,
        comm=comm
    )

    rotation = np.load(COMPRESSION_FILE)

    smap_file = params['input_catalogue'] \
        + "-(map={},fsky={:.2f},knots=[{},{}],rsd={},theta).npy".format(
            'spherical', params['fsky'],
            params['kmin'], params['khyb'],
            params['rsd']
        )
    smap_data = np.load(SMAP_PATH/smap_file).item()

    cmap_file = params['input_catalogue'] \
        + "-(map={},fsky={:.2f},knots=[{},{}],orders={},rsd={},theta).npy"\
            .format(
                'cartesian', params['fsky'], params['khyb'], params['kmax'],
                str(params['multipoles']).replace(", ", ","), params['rsd']
            )
    cmap_data = np.load(CMAP_PATH/cmap_file).item()

    windowed_power_model.wavenumbers = cmap_data['k']

    output_data = defaultdict(list)
    file_suffices = [".txt"] if "halos" not in params['input_catalogue'] \
        else ["L.txt", "R.txt"]

    for file_suffix in file_suffices:
        # Load map data.
        spherical_data = SphericalArray.build(
            disc=disc, filling=smap_data[file_suffix]
        )
        cartesian_data = CartesianArray(
            filling=cmap_data[file_suffix],
            coord_key='k',
            variable_key_root='power_'
        )

        # Construct spherical likelihood.
        sph_likelihood_kwargs = dict(
            two_point_model=two_point_model,
            spherical_data=spherical_data,
            mean_number_density=params['nbar'],
            contrast=params['contrast'],
            pivot=params['spherical_pivot'],
            breakdown=params['breakdown'],
            independence=params['sph_mode_independence'],
            exclude_degrees=params['exclude_degrees'],
            rotation=rotation,
            logger=logger,
            comm=comm,
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
            mean_number_density=params['nbar'],
            contrast=params['contrast'],
            pivot=params['cartesian_pivot'],
            orders=params['multipoles'],
            num_covar_sample=params['num_cov_est'],
            logger=logger,
            comm=comm,
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

    if comm.rank == 0:
        np.save(PATHOUT/script_name/filename, results)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    parsed_params = parse_external_args()
    params, tag = initialise()
    output = process()
    finalise(output, tag)
