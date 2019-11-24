"""Sample the hybrid likelihood from simulations.

"""
import warnings
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.lab import CSVCatalog, ConvolvedFFTPower, cosmology
from scipy.interpolate import interp1d

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import (
    DiscreteSpectrum,
    CartesianArray,
    SphericalArray,
)
from harmonia.collections import confirm_directory_path, format_float
from harmonia.cosmology import fiducial_cosmology, modified_power_spectrum
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import (
    TwoPointFunction,
    WindowedPowerSpectrum,
    WindowCorrelation,
)
from hybrid_likelihoods import (
    cartesian_map_log_likelihood as cartesian_likelihood,
    spherical_map_log_likelihood as spherical_likelihood,
)

PK_FILENAME = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"
WINDOW_POLES_FILEROOT = "window_multipoles"
WINDOW_CORRL_FILEROOT = "{}{}"
COUPLINGS_FILEROOT = "{}{}"
CATALOGUE_HEADINGS = ["x", "y", "z", "vx", "vy", "vz", "mass"]


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

    # Likelihood set-up
    ini_params['map'] = params.map
    ini_params['sample_parameter'] = params.sample_param
    ini_params['fixed_parameter'] = params.fixed_param
    ini_params['prior_range'] = params.prior_range
    ini_params['num_sample'] = params.num_sample

    ini_params['ksplit'] = params.ksplit
    ini_params['kmax'] = params.kmax
    ini_params['breakdown'] = params.breakdown

    # Data and modelling set-up
    ini_params['spherical_pivot'] = params.spherical_pivot
    ini_params['cartesian_pivot'] = params.cartesian_pivot

    ini_params['contrast'] = params.contrast
    ini_params['boxsize'] = params.boxsize
    ini_params['dk'] = params.dk
    ini_params['input_file'] = params.input_file
    ini_params['mesh'] = params.mesh

    # Cosmology set-up
    ini_params['nbar'] = params.nbar
    ini_params['bias'] = dict(zip(['low', 'high'], params.bias))
    ini_params['fnl'] = dict(zip(['low', 'high'], [params.fnl]*2))

    if params.sample_param == 'bias':
        ini_params['fixed_value'] = ini_params['fnl']
        fixed_tag = 'fnl={}'.format(params.fnl)
    elif params.sample_param == 'fnl':
        ini_params['fixed_value'] = ini_params['bias']
        fixed_tag = 'b1={}'.format(str(params.bias).replace(" ", ""))

    # External set-up
    ini_params['matter_power_spectrum'] = interp1d(
        *np.loadtxt(
            "".join([PATHIN, script_name, "/", PK_FILENAME, ".txt"]),
            unpack=True
        ),
        assume_sorted=True
    )

    if params.load_couplings:
        ini_params['external_couplings'] = np.load(
            "".join(
                [
                    PATHIN, script_name, "/",
                    COUPLINGS_FILEROOT.format(
                        format_float(params.boxsize/2, 'intdot'),
                        format_float(params.kmax, 'sci'),
                    ),
                    ".txt"
                ]
            )
        ).item()
    else:
        ini_params['external_couplings'] = None

    ini_info = "map={},prior={},pivots={},ks={},km={},nbar={},{}".format(
        params.map,
        str(params.prior_range).replace(" ", ""),
        str(
            [params.spherical_pivot, params.cartesian_pivot]
        ).replace(" ", ""),
        format_float(ini_params['ksplit'], 'sci'),
        format_float(ini_params['kmax'], 'sci'),
        format_float(ini_params['nbar'], 'sci'),
        fixed_tag,
    )

    for param_name, param_value in ini_params.items():
        globals()[param_name] = param_value

    return ini_params, ini_info


def process(runtime_info):
    """Process program.

    Parameters
    ----------
    runtime_info : str
        Runtime information.

    Returns
    -------
    dict
        Program output.

    """
    print(runtime_info + "\n")

    disc = DiscreteSpectrum(boxsize/2, 'dirichlet', ksplit)

    window_multipoles = np.load('{}.npy'.format(WINDOW_POLES_FILEROOT)).item()

    windowed_power_model = WindowedPowerSpectrum(
        power_spectrum=matter_power_spectrum,
        cosmo=fiducial_cosmology,
        window=window_multipoles,
    )
    two_point_model = TwoPointFunction(
        disc,
        power_spectrum=matter_power_spectrum,
        cosmo=fiducial_cosmology,
        couplings=external_couplings
    )

    output_data = defaultdict(list)

    sample_points = np.linspace(*prior_range, num=num_sample+1)
    output_data['parameters'].append(sample_points)

    for file_suffix in ["L", "R"]:
        catalogue_path = "{}{}/{}{}.txt".format(
            PATHIN, script_name, input_file, file_suffix
        )

        data_catalogue = load_catalogue_from_file(
            catalogue_path, CATALOGUE_HEADINGS, boxsize
        )
        random_catalogue = RandomCatalogue(contrast*nbar, boxsize)

        spherical_map = SphericalMap(
            disc, data_catalogue, rand=random_catalogue,
            mean_density_data=nbar,
            mean_density_rand=contrast*nbar
        )

        mesh = spherical_map.pair.to_mesh(
            Nmesh=mesh, resampler='tsc', compensated=True
        )

        cartesian_power = ConvolvedFFTPower(
            mesh, poles=[0], kmin=ksplit, kmax=kmax
        ).poles

        overdensity_field = SphericalArray.build(
            filling=spherical_map.density_constrast(),
            disc=disc
        )

        spherical_likelihood_args = (
            nbar, two_point_model, overdensity_field, spherical_pivot
        )
        spherical_likelihood_kwargs = dict(
            breakdown=breakdown,
            independence=True,
            contrast=contrast,
        )
        spherical_likelihood_kwargs.update(
            {
                sample_parameter: sample_points,
                fixed_parameter: fixed_value['low'],
            }
        )

        output_data['spherical_likelihood'].append(
            [
                spherical_likelihood(
                    *spherical_likelihood_args,
                    **spherical_likelihood_kwargs
                )
            ]
        )

        compressed_multipoles = CartesianArray(
            filling={
                'k': cartesian_power['k'],
                'power_0': cartesian_power['power_0'].real,
            },
            coord_key='k',
            var_key_root='power_'
        )

        fake_fiducial_model = modified_power_spectrum(
            f_nl=fnl['high'], b_1=bias['high'],
            cosmo=fiducial_cosmology,
            power_spectrum=matter_power_spectrum
        )(cartesian_power['k'])

        window_corr_modeller = WindowCorrelation(
            CartesianArray(
                filling={
                    'k': cartesian_power['k'],
                    'power_0': fake_fiducial_model,
                },
                coord_key='k',
                var_key_root='power_'
            ),
            cartesian_pivot
        )
        window_corr_modeller.window_correlation = np.diag(
            np.sqrt(2/cartesian_power['modes']) * fake_fiducial_model
        )

        cartesian_likelihood_args = (
            nbar,
            windowed_power_model,
            compressed_multipoles,
            window_corr_modeller,
            cartesian_pivot,
        )
        cartesian_likelihood_kwargs = dict(
            contrast=contrast,
        )
        cartesian_likelihood_kwargs.update(
            {
                sample_parameter: sample_points,
                fixed_parameter: fixed_value['high'],
            }
        )

        output_data['cartesian_likelihood'].append(
            [
                cartesian_likelihood(
                    *cartesian_likelihood_args,
                    **cartesian_likelihood_kwargs
                )
            ]
        )

        output_data['data_vector'].append(np.concatenate((
            overdensity_field.unfold(spherical_pivot, return_only='data'),
            compressed_multipoles.unfold(cartesian_pivot, return_only='data'),
        )))

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
    output = process(information)
    finalise(output, information)
