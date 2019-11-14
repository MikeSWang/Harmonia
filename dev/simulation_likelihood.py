"""Sample spherical likelihood from simulations.

"""
import warnings
from collections import defaultdict
from pprint import pprint

import numpy as np
from nbodykit.lab import CSVCatalog, ConvolvedFFTPower, FFTPower
from scipy.interpolate import interp1d

from likelihood_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import confirm_directory_path, format_float
from harmonia.cosmology import fiducial_cosmology
from harmonia.mapper import RandomCatalogue, SphericalMap
from harmonia.reader import TwoPointFunction
from hybrid_likelihoods import (
    cartesian_map_likelihood,
    spherical_map_likelihood,
)

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

    # Likelihood set-up
    ini_params['map'] = params.map
    ini_params['prior_range'] = params.prior_range
    ini_params['num_sample'] = params.num_sample
    ini_params['breakdown'] = params.breakdown
    ini_params['kmax'] = params.kmax
    ini_params['ksplit'] = params.ksplit
    ini_params['pivot'] = params.pivot

    # Catalogue set-up
    ini_params['contrast'] = params.contrast
    ini_params['dk'] = params.dk
    ini_params['boxsize'] = params.boxsize
    ini_params['input_file'] = params.input_file
    ini_params['mesh_cal'] = params.mesh_cal

    # Cosmology set-up
    ini_params['nbar'] = params.nbar
    ini_params['bias'] = params.bias
    ini_params['fnl'] = params.fnl

    if params.growth_rate is None:
        ini_params['growth_rate'] = None
        rsd_tag = 'none'
    else:
        ini_params['growth_rate'] = params.growth_rate
        rsd_tag = "{:.2f}".format(ini_params['growth_rate'])

    if params.likelihood == 'bias':
        ini_params['param'] = 'bias'
        ini_params['fixed'] = {'f_nl': ini_params['fnl']}
        fixed_tag = 'fnl={}'.format(ini_params['fnl'])
    elif params.likelihood == 'fnl':
        ini_params['param'] = 'f_nl'
        ini_params['fixed'] = {'bias': ini_params['bias']}
        fixed_tag = 'b1={}'.format(ini_params['bias'])

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

    ini_info = "map={},prior={},pivot={},kmax={},nbar={},{},f0={}".format(
        ini_params['map'],
        str(ini_params['prior_range']).replace(" ", ""),
        ini_params['pivot'],
        format_float(ini_params['kmax'], 'sci'),
        format_float(ini_params['nbar'], 'sci'),
        fixed_tag,
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
    print(runtime_info + "\n")

    disc = DiscreteSpectrum(
        runtime_params['boxsize']/2, 'Dirichlet', runtime_params['ksplit']
    )

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

        data_catalogue = load_catalogue(
            catalogue_path, runtime_params['boxsize']
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
        spherical_likelihood_kwargs.update(runtime_params['fixed'])

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
        cartesian_likelihood_kwargs.update(runtime_params['fixed'])

        output_data['cartesian_likelihood'].append(
            [
                cartesian_map_likelihood(
                    *cartesian_likelihood_args,
                    **cartesian_likelihood_kwargs
                )
            ]
        )

        output_data['data_vector'] = np.concatenate((
            overdensity_field.unfold(
                runtime_params['pivot'],
                return_only='data'
            ),
            compressed_data['Pk']
        ))

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
