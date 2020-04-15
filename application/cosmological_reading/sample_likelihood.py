"""Sample cosmological likelihoods for bias and primoridial
non-Gaussainity (PNG) on a mesh grid.

"""
import os
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product

import numpy as np
from mpi4py import MPI

try:
    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.algorithms import CartesianArray, SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import (
        CartesianMultipoles,
        Couplings,
        LogLikelihood,
        SphericalCorrelator,
    )
    from harmonia.surveyor.synthesis import \
        CovarianceEstimator, generate_compression_matrix
    from harmonia.utils import mpi_compute
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.algorithms import CartesianArray, SphericalArray
    from harmonia.cosmology import BaseModel
    from harmonia.reader import (
        CartesianMultipoles,
        Couplings,
        LogLikelihood,
        SphericalCorrelator,
    )
    from harmonia.surveyor.synthesis import \
        CovarianceEstimator, generate_compression_matrix
    from harmonia.utils import mpi_compute

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@display_args(logger=logger, comm=comm)
def initialise_parameters():
    """Initialise the program parameters passed from ``stdin``.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed parameter namespace.

    """
    parser = ArgumentParser()

    parser.add_argument(
        '--likelihood', type=str, choices=['cartesian', 'spherical', 'hybrid'],
        help="map likelihood to use"
    )
    parser.add_argument(
        '--no-monopole', action='store_true',
        help="exclude monopoles from spherical likelihood evaluation"
    )
    parser.add_argument(
        '--orders', type=int, nargs='+', default=[0],
        help=(
            "orders of power spectrum multipoles "
            "for Cartesian likelihood evaluation"
        )
    )
    parser.add_argument(
        '--bias', type=float, nargs='+',
        help="fixed bias or bias sampling range and sample intervals"
    )
    parser.add_argument(
        '--png', type=float, nargs='+',
        help="fixed PNG or PNG sampling range and sample intervals"
    )

    parser.add_argument(
        '--map-dir', type=str, default='raw/catalogue_maps',
        help="catalogue map directory relative to 'storage/'"
    )
    parser.add_argument(
        '--map-source-name', type=str, default='',
        help="catalogue source name"
    )
    parser.add_argument(
        '--map-file-extension', type=str, nargs=2, default=['.npz', '.npz'],
        help="catalogue map file extension"
    )

    parser.add_argument(
        '--kmin', type=float, default=None,
        help="minimum wavenumber of the hybrid map"
    )
    parser.add_argument(
        '--khyb', type=float, default=None,
        help="hybridisation wavenumber of the hybrid map"
    )
    parser.add_argument(
        '--kmax', type=float, default=0.1,
        help="maximum wavenumber of the hybrid map"
    )
    parser.add_argument(
        '--rsd', action='store_true',
        help="indicating whether redshift-space distortions are present"
    )

    parser.add_argument(
        '--density', type=float, default=None,
        help="mean number density of the catalogue"
    )
    parser.add_argument(
        '--contrast', type=float, default=10.,
        help="this times higher mean number density for the random catalogue"
    )
    parser.add_argument(
        '--redshift', type=float, default=1.,
        help="this times higher mean number density for the random catalogue"
    )

    parser.add_argument(
        '--cosmology-file', type=str, required=True,
        help=(
            "cosmological parameter file "
            "relative to 'storage/external/cosmology/'"
        )
    )
    parser.add_argument(
        '--mask-multipole-file', type=str, default=None,
        help=(
            "mask function multipole file "
            "relative to 'storage/processed/survey_products/'"
        )
    )
    parser.add_argument(
        '--window-multipole-file', type=str, default=None,
        help=(
            "window function multipole file "
            "relative to 'storage/processed/survey_products/'"
        )
    )
    parser.add_argument(
        '--couplings-file', type=str, default=None,
        help=(
            "spherical coupling multipole file "
            "relative to 'storage/processed/survey_products/'"
        )
    )
    parser.add_argument(
        '--covariance-file', type=str, default=None,
        help=(
            "power spctrum multipole covariance estimate file "
            "relative to 'storage/processed/survey_products/'"
        )
    )

    parser.add_argument(
        '--selection-tag', type=str, default=None,
        help="tag indicating the survey selection function"
    )
    parser.add_argument(
        '--mask-tag', type=str, default=None,
        help="tag indicating the survey mask function"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def setup_likelihood():
    """Set up log-likelihood function.

    Returns
    -------
    :class:`~harmonia.reader.likelihoods.LogLikelihood`
        Log-likelihood function (object).

    """
    simulation_cosmo = BaseModel(cosmo_dir/params.cosmology_file, comm=comm)

    growth_rate = None if params.rsd else 0.

    if params.likelihood in ['spherical', 'hybrid']:
        spherical_data_file = data_dir/params.map_dir/input_filename.format(
            "spherical", params.kmin, params.khyb, None
        )
        spherical_data = SphericalArray.load(
            spherical_data_file.with_suffix(
                spherical_data_file.suffix + params.map_file_extension[0]
            )
        )

        if params.couplings_file is not None:
            couplings = Couplings.load(
                survey_product_dir/
                params.couplings_file.format(
                    params.khyb, params.mask_tag, params.selection_tag
                )
            )

        spherical_model = SphericalCorrelator(
            spherical_data.disc, params.redshift,
            cosmo=simulation_cosmo, growth_rate=growth_rate,
            couplings=couplings, comm=comm
        )
    else:
        spherical_data = None
        spherical_model = None

    if params.likelihood in ['cartesian', 'hybrid']:
        cartesian_data_file = data_dir/params.map_dir/input_filename.format(
            "cartesian", params.khyb, params.kmax, order_tag
        )
        cartesian_data = CartesianArray.load(
            cartesian_data_file.with_suffix(
                cartesian_data_file.suffix + params.map_file_extension[1]
            )
        )

        if params.mask_multipole_file is not None:
            mask_multipoles = np.load(
                survey_product_dir/params.mask_multipole_file
            )
        else:
            mask_multipoles = None

        if params.window_multipole_file is not None:
            window_multipoles = CartesianArray.load(
                survey_product_dir/params.window_multipole_file
            )
        else:
            window_multipoles = None

        cartesian_model = CartesianMultipoles(
            cartesian_data.attrs['wavenumbers'], params.redshift,
            cosmo=simulation_cosmo, growth_rate=growth_rate,
            mask_multipoles=mask_multipoles,
            window_multipoles=window_multipoles
        )

        if params.covariance_file is not None:
            covariance_estimator = CovarianceEstimator.load(
                survey_product_dir/params.covariance_file
            )
            mode_counts = None
        else:
            covariance_estimator = None
            mode_counts = cartesian_data.attrs['mode_counts']
    else:
        cartesian_data = None
        covariance_estimator = None
        mode_counts = None
        cartesian_model = None

    return LogLikelihood(
        spherical_pivot=spherical_pivot, cartesian_pivot=cartesian_pivot,
        spherical_data=spherical_data, cartesian_data=cartesian_data,
        covariance_estimator=covariance_estimator,
        mode_counts=mode_counts,
        base_spherical_model=spherical_model,
        base_cartesian_model=cartesian_model,
        nbar=params.density, contrast=params.contrast,
        comm=comm
    )


def evaluate_likelihood():
    """Evaluate the log-likelihood function.

    Returns
    -------
    :class:`collections.defaultdict`
        Log-likelihood evaluated at sample points.

    """
    num_sample_point = 1

    if len(params.bias) == 1:
        b_1 = params.bias
    elif len(params.bias) == 3:
        b_1 = np.linspace(
            params.bias[0], params.bias[1], num=int(params.bias[2])+1
        ).tolist()
        num_sample_point *= (int(params.bias[2]) + 1)
    else:
        raise ValueError(
            "Input bias must be a scalar (length 1) or a range (length 3)."
        )

    if len(params.png) == 1:
        f_nl = params.png
    elif len(params.png) == 3:
        f_nl = np.linspace(
            params.png[0], params.png[1], num=int(params.png[2])+1
        ).tolist()
        num_sample_point *= (int(params.png[2]) + 1)
    else:
        raise ValueError(
            "Input PNG must be a scalar (length 1) or a range (length 3)."
        )

    if params.likelihood in ['spherical', 'hybrid']:
        # Radialise to speed up process when survey is trivial.
        radialise = (
            params.mask_tag is None or params.mask_tag.startswith('1.')
        ) and (
            params.selection_tag is None
            or params.selection_tag.lower() == 'none'
        )

        exclude_degrees = (0,) if params.no_monopole else ()

        logger.info("Generating a spherical data compression matrix...")
        compression_matrix = generate_compression_matrix(
            {
                'pivot': spherical_pivot,
                'spherical_model': likelihood_function.base_spherical_model,
                'b_1': 2.5,
                'f_nl': 0.,
            },  # fiducial model
            {
                'pivot': spherical_pivot,
                'spherical_model': likelihood_function.base_spherical_model,
                'b_1': 3.,
                'f_nl': 100.,
            } # extremal model
        )
        logger.info("... generated a spherical data compression matrix.")

        spherical_likelihood_kwargs = {
            'exclude_degrees': exclude_degrees,
            'compression_matrix': compression_matrix,
            'radialise': radialise,
        }

    if params.likelihood in ['cartesian', 'hybrid']:
        orders = params.orders

        try:
            # pylint: disable=eval-used
            num_samples = abs(eval(
                re.search('source=(?P<range>.*?),', covariance_file)
                .group('range')
            )) + 1
        except NameError:
            num_samples = None

        cartesian_likelihood_kwargs = {
            'orders': orders,
            'num_samples': num_samples,
        }

        # Temporarily disable `num_samples`.
        cartesian_likelihood_kwargs.update({'num_samples': None})

    sample_points = list(product(b_1, f_nl))
    log_likelihood = defaultdict(list)

    if params.likelihood in ['spherical', 'hybrid']:
        def _eval_likelihood(sample_point):
            return likelihood_function.spherical_map_likelihood(
                *sample_point, **spherical_likelihood_kwargs
            )
        spherical_likelihood = mpi_compute(
            sample_points, _eval_likelihood, comm=comm,
            process_name="spherical likelihood",
            update_rate=len(sample_points)/comm.size
        )
        log_likelihood['spherical_likelihood'].extend(spherical_likelihood)

    if params.likelihood in ['cartesian', 'hybrid']:
        def _eval_likelihood(sample_point):
            return likelihood_function.cartesian_map_likelihood(
                *sample_point, **cartesian_likelihood_kwargs
            )
        cartesian_likelihood = mpi_compute(
            sample_points, _eval_likelihood, comm=comm,
            process_name="Cartesian likelihood",
            update_rate=len(sample_points)/comm.size
        )
        log_likelihood['cartesian_likelihood'].extend(cartesian_likelihood)

    for key, values in log_likelihood.items():
        log_likelihood[key] = np.reshape(values, (len(b_1), len(f_nl)))

    log_likelihood['b_1'] = b_1
    log_likelihood['f_nl'] = f_nl

    return log_likelihood


spherical_pivot, cartesian_pivot = 'natural', 'order'

if __name__ == '__main__':

    params = initialise_parameters()

    order_tag = str(params.orders).replace(" ", "")

    # Set I/O paths.
    cosmo_dir = data_dir/"external"/"cosmology"
    survey_product_dir = data_dir/"processed"/"survey_products"

    input_dir = data_dir/params.map_dir
    input_filename = "catalogue-map-({})".format(",".join([
        "source={}".format(params.map_source_name),
        "map={}", "scale=[{},{}]", "orders={}",
        "rsd={}".format(params.rsd),
        "mask={}".format(params.mask_tag),
        "selection={}".format(params.selection_tag),
    ]))

    output_dir = data_dir/"raw"/"likelihoods"
    output_filename = "likelihood-({})".format(",".join([
        "source={}".format(params.map_source_name),
        "map={}".format(params.likelihood),
        "scale=[{},{},{}]".format(params.kmin, params.khyb, params.kmax),
        "orders={}".format(order_tag),
        "excl_monop={}".format(params.no_monopole),
        "rsd={}".format(params.rsd),
        "mask={}".format(params.mask_tag),
        "selection={}".format(params.selection_tag),
    ]))

    confirm_directory(output_dir)

    # Sample likelihood on a mesh grid and export results.
    likelihood_function = setup_likelihood()
    log_likelihoods = evaluate_likelihood()
    np.savez(output_dir/output_filename, **log_likelihoods)
