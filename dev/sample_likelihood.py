"""Sample the hybrid likelihood from simulations.

"""
import warnings
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.lab import CSVCatalog, ConvolvedFFTPower
from scipy.interpolate import interp1d

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path, format_float
from harmonia.cosmology import fiducial_cosmology
from harmonia.mapper import (
    RandomCatalogue,
    SphericalMap,
    load_catalogue_from_file,
)
from harmonia.reader import TwoPointFunction, WindowedPowerSpectrum
from hybrid_likelihoods import (
    cartesian_map_log_likelihood as cartesian_likelihood,
    spherical_map_log_likelihood as spherical_likelihood,
)

PK_FILENAME = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"
COUPLINGS_FILEROOT = "{}{}"
WINDOW_POLES_FILEROOT = "{}{}"
WINDOW_CORRL_FILEROOT = "{}{}"
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
    ini_params['fnl'] = params.fnl

    if params.sample_param == 'bias':
        fixed_tag = 'fnl={}'.format(params.fnl)
    elif params.sample_param == 'fnl':
        fixed_tag = 'b1={}'.format(str(params.prior_range).replace(" ", ""))

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

    ini_info =\
        "map={},prior={},pivots={},ks={},km={},nbar={},{}".format(
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

    window_multipoles = np.load(WINDOW_POLES_FILEROOT).item()
    window_corrmat = np.load(WINDOW_CORRL_FILEROOT).item()

    windowed_power_model = WindowedPowerSpectrum()

    two_point_model = TwoPointFunction(
        disc,
        f_0=runtime_params['growth_rate'],
        power_spectrum=runtime_params['matter_power_spectrum'],
        cosmo=fiducial_cosmology,
        couplings=runtime_params['external_couplings']
    )

    output_data = defaultdict(list)

    sample_parameters = np.linspace(
        *runtime_params['prior_range'],
        num=runtime_params['num_sample']+1
    )

    output_data['parameters'].append(sample_parameters)

    for file_suffix in ["L", "R"]:
        catalogue_path = "{}{}/{}{}.txt".format(
            PATHIN, script_name, runtime_params['input_file'], file_suffix
        )

        data_catalogue = load_catalogue_from_file(
            catalogue_path, CATALOGUE_HEADINGS, runtime_params['boxsize']
        )

        # random_catalogue = RandomCatalogue(
        #     runtime_params['contrast'] * runtime_params['nbar'],
        #     runtime_params['boxsize']
        # )

        mesh = data_catalogue.to_mesh(
            Nmesh=runtime_params['mesh_cal'],
            resampler='tsc',
            compensated=True
        )

        spherical_map = SphericalMap(
            disc,
            data_catalogue,
            mean_density_data=runtime_params['nbar']
        )

        # spherical_map = SphericalMap(
        #     disc,
        #     data_catalogue,
        #     rand=random_catalogue,
        #     mean_density_data=runtime_params['nbar'],
        #     mean_density_rand=\
        #         runtime_params['contrast']*runtime_params['nbar']
        # )

        # mesh = spherical_map.pair.to_mesh(
        #     Nmesh=runtime_params['mesh_cal'],
        #     resampler='tsc',
        #     compensated=True
        # )

        cartesian_power = FFTPower(
            mesh,
            mode='1d',
            kmin=runtime_params['ksplit'],
            kmax=runtime_params['kmax']
        ).power

        # cartesian_power = ConvolvedFFTPower(
        #     mesh,
        #     poles=[0],
        #     kmin=runtime_params['ksplit'],
        #     kmax=runtime_params['kmax']
        # ).poles

        overdensity_field = SphericalArray.build(
            filling=spherical_map.density_constrast(),
            disc=disc
        )

        spherical_likelihood_args = (
            sample_parameters,
            runtime_params['param'],
            overdensity_field,
            two_point_model,
            runtime_params['pivot'],
            runtime_params['nbar'],
        )
        spherical_likelihood_kwargs = dict(
            breakdown=runtime_params['breakdown'],
            remove_degrees=(),
            mode_indices=index_vector,
            independence=True,
        )
        spherical_likelihood_kwargs.update(runtime_params['fixed']['low'])

        output_data['spherical_likelihood'].append(
            [
                spherical_map_likelihood(
                    *spherical_likelihood_args,
                    **spherical_likelihood_kwargs
                )
            ]
        )

        compressed_data = {
            'k': cartesian_power['k'],
            'Nk': cartesian_power['modes'],
            'Pk': cartesian_power['power'].real,
            'Pshot': cartesian_power.attrs['shotnoise'],
        }

        pprint(cartesian_power['k'])
        pprint(cartesian_power['modes'])
        # output_data['k'].append(cartesian_power['k'])
        # output_data['Nk'].append(cartesian_power['modes'])

        cartesian_likelihood_args = (
            sample_parameters,
            runtime_params['param'],
            compressed_data,
            runtime_params['nbar'],
        )
        cartesian_likelihood_kwargs = dict(
            cosmo=fiducial_cosmology,
            contrast=runtime_params['contrast'],
            power_spectrum=runtime_params['matter_power_spectrum'],
        )
        cartesian_likelihood_kwargs.update(runtime_params['fixed']['high'])

        output_data['cartesian_likelihood'].append(
            [
                cartesian_map_likelihood(
                    *cartesian_likelihood_args,
                    **cartesian_likelihood_kwargs
                )
            ]
        )

        output_data['data_vector'].append(np.concatenate((
            overdensity_field.unfold(
                runtime_params['pivot'],
                return_only='data'
            ),
            compressed_data['Pk']
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
    output = process(parameters, information)
    finalise(output, information)
