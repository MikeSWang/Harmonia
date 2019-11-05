"""Sample spherical likelihood from simulations.

"""
import warnings

import numpy as np
from nbodykit.lab import CSVCatalog, ConvolvedFFTPower
from scipy.interpolate import interp1d

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path, format_float
from harmonia.cosmology import fiducial_cosmology
from harmonia.mapper import RandomCatalogue, SphericalMap
from harmonia.reader import TwoPointFunction
from cartesian_likelihood import cartesian_map_likelihood
from spherical_likelihood import spherical_map_likelihood

PK_FILENAME = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


def load_catalogue(catalogue_path, boxsize):
    """Load catalogue from files.

    Parameters
    ----------
    catalogue_path : str
        Catalogue file path.
    boxsize : float
        Catalogue box size.

    Returns
    -------
    catalogue : :class:`nbodykit.base.catalog.CatalogSource`
        Catalogue object.

    """
    catalogue = CSVCatalog(catalogue_path, CATALOGUE_HEADINGS)

    catalogue.attrs['BoxSize'] = boxsize
    catalogue['Position'] = \
        catalogue['x'][:, None] * [1, 0, 0] \
        + catalogue['y'][:, None] * [0, 1, 0] \
        + catalogue['z'][:, None] * [0, 0, 1]

    return catalogue


def initialise():
    """Initialise program.

    Returns
    -------
    ini_params : dict
        Initialised parameters.
    ini_info : str
        Initialisation information.

    """
    ini_params = {}

    # Runtime set-up
    ini_params['pivot'] = params.pivot
    ini_params['kmax'] = params.kmax

    # Likelihood set-up
    ini_params['likelihood'] = params.likelihood
    ini_params['prior_range'] = params.prior_range
    ini_params['num_sample'] = params.num_sample
    ini_params['breakdown'] = params.breakdown

    # Catalogue set-up
    ini_params['contrast'] = params.contrast
    ini_params['boxsize'] = params.boxsize
    ini_params['input_file'] = params.input_file
    ini_params['dk'] = params.dk
    ini_params['mesh_cal'] = params.mesh_cal

    # Cosmology set-up
    ini_params['nbar'] = params.nbar
    ini_params['bias'] = params.bias

    if params.growth_rate is None:
        ini_params['growth_rate'] = None
        rsd_tag = 'none'
    else:
        ini_params['growth_rate'] = params.growth_rate
        rsd_tag = "{:.2f}".format(ini_params['growth_rate'])

    ini_params['matter_power_spectrum'] = interp1d(
        *np.loadtxt(
            "".join([PATHIN, script_name, "/", PK_FILENAME, ".txt"])
        ).T,
        assume_sorted=True
    )

    if params.load_couplings:
        ini_params['external_couplings'] = np.load(
            f"{PATHIN}{script_name}/"
            "nbodymod_twopt-(rmax={},kmax={})-couplings.npy"
            .format(
                format_float(params.boxsize/2, 'intdot'),
                format_float(params.kmax, 'sci'),
            )
        ).item()
    else:
        ini_params['external_couplings'] = None

    ini_info = "map={},prior={},pivot={},kmax={},nbar={},b1={},f0={}".format(
        ini_params['likelihood'],
        str(ini_params['prior_range']).replace(" ", ""),
        ini_params['pivot'],
        format_float(ini_params['kmax'], 'sci'),
        format_float(ini_params['nbar'], 'sci'),
        ini_params['bias'],
        rsd_tag,
    )

    return ini_params, ini_info


def process(runtime_params, runtime_info):
    """Process program.

    Parameters
    ----------
    runtime_params : dict
        Runtime parameters.
    runtime_info : str
        Runtime information.

    Returns
    -------
    dict
        Program output.

    """
    print(runtime_info)

    disc = DiscreteSpectrum(
        runtime_params['boxsize']/2, 'Dirichlet', runtime_params['kmax']
    )

    if runtime_params['likelihood'] == 'spherical':
        index_vector = SphericalArray\
            .build(disc=disc)\
            .unfold(runtime_params['pivot'], return_only='index')

        two_point_model = TwoPointFunction(
            disc,
            f_0=runtime_params['growth_rate'],
            power_spectrum=runtime_params['matter_power_spectrum'],
            cosmo=fiducial_cosmology,
            couplings=runtime_params['external_couplings']
        )
        log_likelihood = spherical_map_likelihood
    elif runtime_params['likelihood'] == 'cartesian':
        log_likelihood = cartesian_map_likelihood

    sample_parameters = np.linspace(
        *runtime_params['prior_range'],
        num=runtime_params['num_sample']+1
    )

    sample_values = []
    for file_suffix in ["L", "R"]:
        catalogue_path = "{}{}/{}{}.txt".format(
            PATHIN, script_name, runtime_params['input_file'], file_suffix
        )
        catalogue = load_catalogue(catalogue_path, runtime_params['boxsize'])

        if runtime_params['likelihood'] == 'spherical':
            spherical_map = SphericalMap(
                disc, catalogue, mean_density_data=runtime_params['nbar']
            )
            overdensity_field = SphericalArray.build(
                filling=spherical_map.density_constrast(), disc=disc
            )

            log_likelihood_args = (
                sample_parameters,
                'f_nl', # 'bias', #
                overdensity_field,
                two_point_model,
                runtime_params['pivot'],
                runtime_params['nbar'],
            )
            log_likelihood_kwargs = dict(
                bias=runtime_params['bias'], # f_nl=None, #
                breakdown=runtime_params['breakdown'],
                remove_degrees=(),
                mode_indices=index_vector,
            )
        elif runtime_params['likelihood'] == 'cartesian':

            random_catalogue = RandomCatalogue(
                runtime_params['contrast'] * runtime_params['nbar'],
                runtime_params['boxsize']
            )

            spherical_map = SphericalMap(
                disc,
                catalogue,
                rand=random_catalogue,
                mean_density_data=runtime_params['nbar'],
                mean_density_rand=\
                    runtime_params['contrast']*runtime_params['nbar']
            )

            mesh = spherical_map.pair.to_mesh(
                Nmesh=runtime_params['mesh_cal'],
                resampler='tsc',
                compensated=True
            )
            cartesian_power = ConvolvedFFTPower(
                mesh,
                poles=[0],
                dk=runtime_params['dk'],
                kmax=runtime_params['kmax']
            ).poles

            compressed_data = {
                'k': cartesian_power['k'],
                'Nk': cartesian_power['modes'],
                'Pk': cartesian_power['power_0'].real,
                'Pshot': cartesian_power.attrs['shotnoise'],
            }

            log_likelihood_args = (
                sample_parameters,
                'f_nl',  # 'bias', #
                compressed_data,
                runtime_params['nbar'],
            )
            log_likelihood_kwargs = dict(
                cosmo=fiducial_cosmology,
                bias=runtime_params['bias'],  # f_nl=None, #
                contrast=runtime_params['contrast'],
                power_spectrum=runtime_params['matter_power_spectrum'],
            )

        sample_values.append(
            log_likelihood(*log_likelihood_args, **log_likelihood_kwargs)
        )

    output_data = {
        'parameters': [sample_parameters],
        'likelihood': [sample_values],
    }

    return output_data


def finalise(output_data, output_info):
    """Finalise program.

    Parameters
    ----------
    output_data : dict
        Output data.
    output_info : dict
        Program output information.

    """
    file_tag = "-({})".format(output_info)

    base_path = f"{PATHOUT}{script_name}/"
    assert confirm_directory_path(base_path)

    file_path = base_path + f"{parameters['input_file']}{file_tag}.npy"

    np.save(file_path, output_data)


if __name__ == '__main__':

    warnings.simplefilter("ignore", UserWarning)

    parameters, information = initialise()
    output = process(parameters, information)
    finalise(output, information)
