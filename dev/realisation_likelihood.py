"""Sample spherical likelihood from realisations.

"""
import warnings

import numpy as np
from nbodykit.lab import cosmology

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path, format_float
from harmonia.cosmology import fiducial_cosmology, fiducial_distance
from harmonia.mapper import LogNormalCatalogue, NBKCatalogue, SphericalMap
from harmonia.reader import TwoPointFunction
from spherical_likelihood import spherical_map_likelihood as log_likelihood

GEN_CATALOGUE = {
    'lognormal': LogNormalCatalogue,
    'nbodykit': NBKCatalogue,
}


def initialise():
    """Initialise program.

    Returns
    -------
    ini_params : dict
        Initialised parameters.
    ini_tag : str
        Initialisation information.

    """
    ini_params = {}

    # Runtime set-up
    ini_params['pivot'] = params.pivot
    ini_params['kmax'] = params.kmax

    ini_params['niter'] = params.niter
    ini_params['prog_id'] = params.prog_id

    # Likelihood set-up
    ini_params['prior_range'] = params.prior_range
    ini_params['num_sample'] = params.num_sample
    ini_params['breakdown'] = params.breakdown

    # Catalogue set-up
    if params.generator.lower().startswith('l'):
        ini_params['generator'] = "lognormal"
    elif params.generator.lower().startswith('n'):
        ini_params['generator'] = "nbodykit"

    ini_params['rsd_flag'] = params.rsd
    ini_params['rmax'] = fiducial_distance(params.zmax)
    ini_params['mesh_gen'] = params.mesh_gen

    # Cosmology set-up
    ini_params['nbar'] = params.nbar
    ini_params['bias'] = params.bias

    if ini_params['rsd_flag']:
        ini_params['growth_rate'] = \
            fiducial_cosmology.scale_independent_growth_rate(params.redshift)
        rsd_tag = "{:.2f}".format(ini_params['growth_rate'])
    else:
        ini_params['growth_rate'] = None
        rsd_tag = 'none'

    ini_params['matter_power_spectrum'] = cosmology.LinearPower(
        fiducial_cosmology, redshift=params.redshift
    )

    if params.load_couplings:
        ini_params['external_couplings'] = np.load(
            f"{PATHIN}{script_name}/"
            "predict_twopt-(rmax={},kmax={})-couplings.npy"
            .format(
                format_float(ini_params['rmax'], 'intdot'),
                format_float(ini_params['kmax'], 'sci'),
            )
        ).item()
    else:
        ini_params['external_couplings'] = None

    ini_tag = ",".join(
        [
            "-(prior={},pivot={},kmax={}".format(
                str(ini_params['prior_range']).replace(" ", ""),
                ini_params['pivot'],
                format_float(ini_params['kmax'], 'sci'),
            ),
            "gen={}".format(ini_params['generator']),
            "nbar={},b1={},f0={}".format(
                format_float(ini_params['nbar'], 'sci'),
                format_float(ini_params['bias'], 'decdot'),
                rsd_tag,
            ),
            "rmax={},mesh={}".format(
                format_float(ini_params['rmax'], 'intdot'),
                ini_params['mesh_gen'],
            ),
            "iter={})-[{}]".format(ini_params['niter'], ini_params['prog_id']),
        ]
    )

    return ini_params, ini_tag


def process(runtime_params, runtime_tag):
    """Process program.

    Parameters
    ----------
    runtime_params : dict
        Runtime parameters.
    runtime_tag : str
        Runtime tag.

    Returns
    -------
    output_data : dict
        Program output.

    """
    print(runtime_tag.strip("-").lstrip("(").rstrip(")"))

    disc = DiscreteSpectrum(
        runtime_params['rmax'], 'Dirichlet', runtime_params['kmax']
    )

    index_vector = SphericalArray\
        .build(disc=disc)\
        .unfold(runtime_params['pivot'], return_only='index')

    two_point_model = TwoPointFunction(
        disc,
        f_0=runtime_params['growth_rate'],
        cosmo=fiducial_cosmology,
        couplings=runtime_params['external_couplings']
    )

    sample_parameters = np.linspace(
        *runtime_params['prior_range'],
        num=runtime_params['num_sample']+1
    )

    sample_values = []
    for run in range(runtime_params['niter']):
        catalogue = GEN_CATALOGUE[runtime_params['generator']](
            runtime_params['matter_power_spectrum'],
            runtime_params['nbar'],
            bias=runtime_params['bias'],
            boxsize=2*runtime_params['rmax'],
            num_mesh=runtime_params['mesh_gen'],
            add_RSD=runtime_params['rsd_flag']
        )

        spherical_map = SphericalMap(
            disc, catalogue, mean_density_data=runtime_params['nbar']
        )
        overdensity_field = SphericalArray.build(
            filling=spherical_map.density_constrast(), disc=disc
        )

        log_likelihood_args = (
            sample_parameters,
            'f_nl',  #
            overdensity_field,
            two_point_model,
            runtime_params['pivot'],
            runtime_params['nbar'],
        )

        log_likelihood_kwargs = dict(
            bias=runtime_params['bias'],  #
            breakdown=runtime_params['breakdown'],
            remove_degrees=(),
            mode_indices=index_vector,
        )

        sample_values.append(
            log_likelihood(*log_likelihood_args, **log_likelihood_kwargs)
        )

    output_data = {
        'parameters': [sample_parameters],
        'likelihood': [sample_values],
    }

    return output_data


def finalise(output_data, process_tag):
    """Finalise program.

    Parameters
    ----------
    output_data : dict
        Output data.
    program_tag : dict
        Program tag.

    """
    file_tag = "-({})".format(process_tag)

    base_path = f"{PATHOUT}{script_name}/"
    assert confirm_directory_path(base_path)

    file_path = base_path + f"{script_name}{file_tag}.npy"

    np.save(file_path, output_data)


if __name__ == '__main__':

    warnings.simplefilter("ignore", UserWarning)

    parameters, tag = initialise()
    output = process(parameters, tag)
    finalise(output, tag)
