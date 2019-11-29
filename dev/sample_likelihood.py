"""Sample the hybrid likelihood from simulations.

"""
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.lab import ConvolvedFFTPower, Cosmology, cosmology

from likelihood_rc import PATHIN, PATHOUT, parsed_params, script_name
from harmonia.algorithms import (
    CartesianArray,
    DiscreteSpectrum,
    SphericalArray,
)
from harmonia.collections import confirm_directory_path
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import (
    TwoPointFunction,
    WindowCorrelation,
    WindowedPowerSpectrum,
)
from harmonia.reader import \
    cartesian_map_log_likelihood as cartesian_likelihood
from harmonia.reader import \
    spherical_map_log_likelihood as spherical_likelihood

COSMOLOGY_FILE = PATHIN/"cosmology"/"cosmological_parameters.txt"
MASK_MULTIPOLES_FILE = PATHIN/"survey"/"mask_multipoles-1.00sky.npy"
WINDOW_MULTIPOLES_FILE = PATHIN/"survey"/"window_multipoles-1.00sky.npy"
FIDUCIAL_FILE = PATHIN/"survey"/"estimation_fiducial-1.00sky.npy"
COVAR_ESTIMATE_FILE = PATHIN/"survey"/"window_correlation-1.00sky.npy"
COUPLINGS_FILE = PATHIN/"survey"/None

CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]
TO_MESH_KWARGS = dict(resampler='tsc', compensated=True, interlaced=True)

simu_cosmo = None
mask_multipoles = None
window_multipoles = None
window_correlation = None
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

    fixed_tag = ""
    ini_params['fixed'] = {}
    for fixed_par_name, fixed_par_values in eval(parsed_params.fixed_param):
        fixed_tag += "{}=[{},{}],".format(fixed_par_name, *fixed_par_values)
        ini_params['fixed'].update(
            {
                fixed_par_name : dict(zip(['low', 'high'], fixed_par_values))
            }
        )

    sampled_tag = ""
    ini_params['sampled'] = {}
    for samp_par_name, samp_par_range in eval(parsed_params.sample_param):
        sampled_tag += "{}_prior=[{},{}],".format(
            samp_par_name, *samp_par_range
        )
        ini_params['sampled'].update(
            {
                samp_par_name: np.linspace(
                    *samp_par_range, num=ini_params['num_sample']
                )
            }
        )

    ini_tag = "map={},pivots=[{},{}],knots=[{},{}],{},{}".format(
        parsed_params.map,
        parsed_params.spherical_pivot, parsed_params.cartesian_pivot,
        parsed_params.khyb, parsed_params.kmax,
        sampled_tag, fixed_tag
    )

    global simu_cosmo, mask_multipoles, window_multipoles, window_correlation,\
        external_couplings

    with open(COSMOLOGY_FILE) as cosmology_file:
        cosmological_parameters = eval(cosmology_file.read())
        simu_cosmo = Cosmology(
            h=cosmological_parameters['h'],
            Omega0_b=cosmological_parameters['Omega0_b'],
            Omega_cdm=cosmological_parameters['Omega_cdm']
        ).match(cosmological_parameters['sigma8'])

    mask_multipoles = np.load(MASK_MULTIPOLES_FILE).item()
    window_multipoles = np.load(WINDOW_MULTIPOLES_FILE).item()
    external_couplings = None

    window_correlation = WindowCorrelation(
        np.load(FIDUCIAL_FILE).item(), parsed_params.cartesian_pivot
    )
    window_correlation.window_correlation = np.load(COVAR_ESTIMATE_FILE)

    return ini_params, ini_tag


def process():
    """Process program.

    Returns
    -------
    dict
        Program output.

    """
    disc = DiscreteSpectrum(params['boxsize']/2, 'dirichlet', params['khyb'])

    matter_power_spectrum = cosmology.LinearPower(
        simu_cosmo, params['redshift']
    )

    windowed_power_model = WindowedPowerSpectrum(
        redshift=params['redshift'],
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
            catalogue_path, CATALOGUE_HEADINGS, params['boxsize']
        )
        random_catalogue = RandomCatalogue(
            params['contrast']*params['nbar'], params['boxsize']
        )
        spherical_map = SphericalMap(
            disc, data_catalogue, rand=random_catalogue,
            mean_density_data=params['nbar'],
            mean_density_rand=params['contrast']*params['nbar']
        )
        cartesian_mesh = spherical_map.pair.to_mesh(
            Nmesh=params['mesh'], **TO_MESH_KWARGS
        )

        # Compute measurements.
        cartesian_power = ConvolvedFFTPower(
            cartesian_mesh,
            poles=params['multipoles'],
            kmin=params['khyb'],
            kmax=params['kmax']
        ).poles

        valid_bins = ~np.equal(cartesian_power['modes'], 0)

        cartesian_multipoles = {
            var_name: cartesian_power[var_name][valid_bins].real
            for var_name in cartesian_power if 'power_' in var_name
        }
        cartesian_multipoles['k'] = cartesian_power['k'][valid_bins]

        cartesian_data = CartesianArray(
            filling=cartesian_multipoles,
            coord_key='k',
            var_key_root='power_'
        )

        overdensity_field = spherical_map.density_constrast()

        spherical_data = SphericalArray.build(
            disc=disc, filling=overdensity_field
        )

        # Construct spherical likelihood.
        spherical_likelihood_kwargs = dict(
            two_point_model=two_point_model,
            spherical_data=spherical_data,
            nbar=params['nbar'],
            contrast=params['contrast'],
            pivot=params['spherical_pivot'],
            breakdown=params['breakdown'],
            independence=True,
        )
        for par_name, par_values in params['fixed']:
            spherical_likelihood_kwargs.update({par_name: par_values['low']})
        for par_name, par_values in params['sampled']:
            spherical_likelihood_kwargs.update({par_name: par_values})

        output_data['spherical_likelihood'].append(
            [spherical_likelihood(**spherical_likelihood_kwargs)]
        )

        # Construct Cartesian likelihood.
        windowed_power_model.wavenumbers = cartesian_data.coord_array
        cartesian_likelihood_kwargs = dict(
            windowed_power_model=windowed_power_model,
            window_corr_modeller=window_correlation,
            cartesian_data=cartesian_data,
            nbar=params['nbar'],
            contrast=params['contrast'],
            pivot=params['cartesian_pivot'],
            orders=params['multipoles'],
        )
        for par_name, par_values in params['fixed']:
            cartesian_likelihood_kwargs.update({par_name: par_values['low']})
        for par_name, par_values in params['sampled']:
            cartesian_likelihood_kwargs.update({par_name: par_values})

        output_data['cartesian_likelihood'].append(
            [cartesian_likelihood(**cartesian_likelihood_kwargs)]
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
