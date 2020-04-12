"""Synthesise correlation estimates from synthesised random maps.

"""
import os
import re
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from application import application_logger as logger
    from application import data_dir, display_args
    from harmonia.algorithms import CartesianArray, SphericalArray
    from harmonia.surveyor.synthesis import CovarianceEstimator
    from harmonia.utils import covar_to_corr
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import data_dir, display_args
    from harmonia.algorithms import CartesianArray, SphericalArray
    from harmonia.surveyor.synthesis import CovarianceEstimator
    from harmonia.utils import covar_to_corr


@display_args(logger=logger)
def initialise_parameters():
    """Initialise the program parameters passed from ``stdin``.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed parameter namespace.

    """
    parser = ArgumentParser()

    parser.add_argument(
        '--cartesian-only', action='store_true',
        help="do not process spherial maps"
    )

    parser.add_argument(
        '--map-dir', type=str, default='raw/random_maps',
        help="random map directory relative to 'storage/'"
    )
    parser.add_argument(
        '--map-file-extension', type=str, nargs='+', default=['.npz', '.npz'],
        help="random map file extension"
    )
    parser.add_argument(
        '--map-source-root', type=str, default='',
        help="random map filename common root"
    )
    parser.add_argument(
        '--map-source-names', type=str, nargs='+', required=True,
        help="random map filenames without extension"
    )

    parser.add_argument(
        '--pivots', type=str, nargs=2, default=['natural', 'order'],
        help="spherical and Cartesian pivots for map data vectorisation"
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
        '--orders', type=int, nargs='+', default=[0],
        help="power spectrum multipoles of the Cartesian map"
    )

    parser.add_argument(
        '--sky-fraction', type=float, default=None,
        help="sky fraction of the map; only used in lieu of --mask-file"
    )
    parser.add_argument(
        '--mask-file', type=str, default=None,
        help="external 'healpy' mask map"
    )
    parser.add_argument(
        '--selection-cut', type=float, nargs=2, default=None,
        help=(
            "cut selection parameters (low_end, high_end); "
            "only used in lieu of --selection-file"
        )
    )
    parser.add_argument(
        '--selection-distribution', nargs=4, default=None,
        help=(
            "'gaussian' selection parameters (peak, scale, location); "
            "'gamma' selection parameters (peak, scale, shape); "
            "only used in lieu of --selection-file"
        )
    )
    parser.add_argument(
        '--selection-file', type=str, default=None,
        help="external selection funtion samples for comoving distance"
    )

    parser.add_argument(
        '--plot-estimate', action='store_true',
        help="plot covariance matrix estimates"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def write_tags():
    """Write output file tags.

    Returns
    -------
    mask_info, selection_info, order_info, pivot_info : str
        Mask, selection, multipole order and pivot information strings.

    """
    if params.mask_file is not None:
        # Only take mask '(source=<>,...)' information.
        mask_info = re.search(
            "source=(?P<source>?.*?),", params.mask_file
        ).group('source')
    else:
        mask_info = params.sky_fraction

    if params.selection_file is not None:
        # Only take selection '(source=<>,...)' information.
        selection_info = re.search(
            "source=(?P<source>?.*?),", params.selection_file
        ).group('source')
        # Also remove double records if the same as `mask_file` source.
        selection_info = 'same' if selection_info == mask_info \
            else selection_info
    else:
        selection_info = params.selection_distribution or params.selection_cut
        selection_info = str(selection_info).replace(" ", "").replace("'", "")

    order_info = str(params.orders).replace(" ", "")
    pivot_info = str(params.pivots).replace(" ", "").replace("'", "")

    return mask_info, selection_info, order_info, pivot_info


def extract_map_data():
    """Load and process random map data for covariance estimation.

    Returns
    -------
    list, list
        Spherical and Cartesian map data.

    """
    source_names = params.map_source_names.split() \
        if isinstance(params.map_source_names, str) \
        else params.map_source_names

    _spherical_data, _cartesian_data = [], []
    for source_name in source_names:
        cartesian_map_file = input_dir/input_filename.format(
            source_name, 'cartesian', params.khyb, params.kmax, order_tag
        )
        cartesian_map_data = CartesianArray.load(
            cartesian_map_file.with_suffix(
                cartesian_map_file.suffix + params.map_file_extension[1]
            )
        )
        _cartesian_data.append(cartesian_map_data)

        if not params.cartesian_only:
            spherical_map_file = input_dir/input_filename.format(
                source_name, 'spherical', params.kmin, params.khyb, None
            )
            spherical_map_data = SphericalArray.load(
                spherical_map_file.with_suffix(
                    spherical_map_file.suffix + params.map_file_extension[0]
                )
            )
            _spherical_data.append(spherical_map_data)

    return _spherical_data, _cartesian_data


def export_map_statistics():
    """Export and visualise statistical properties of random maps.

    """
    # Export Cartesian covariance estimator.
    cartesian_covariance_estimator = CovarianceEstimator(cartesian_data)

    output_file = output_dir/output_filename.format(
        'cartesian', "[{},{}]".format(params.khyb, params.kmax), order_tag
    )

    cartesian_covariance_estimator.save(output_file)

    cartesian_correlation = covar_to_corr(
        cartesian_covariance_estimator.get_fiducial_covariance(
            params.pivots[1]
        )
    )

    sns.set(style='ticks', font='serif')
    plt.figure("random map correlation estimates")
    sns.heatmap(
        cartesian_correlation, cmap='coolwarm', square=True, vmin=-1., vmax=1.
    )
    if params.plot_estimate:
        plt.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))

    if not params.only_cartesian:
        # Export spherical covariance matrix.
        spherical_data_array = [
            spherical_map_data.vectorise(params.pivots[0])
            for spherical_map_data in spherical_data
        ]

        spherical_covariance_estimate = np.linalg.multi_dot([
            np.conj(spherical_data_array).T, spherical_data_array
        ]) / len(spherical_data_array)

        spherical_correlation_estimate = covar_to_corr(
            spherical_covariance_estimate
        )

        output_file = output_dir/output_filename.format(
            'spherical', "[{},{}]".format(params.kmin, params.khyb), None
        )

        np.save(output_file, spherical_covariance_estimate)

        spherical_correlation = np.zeros_like(
            spherical_correlation_estimate, dtype=float
        )

        spherical_tril = np.tril_indices(len(spherical_correlation))
        spherical_triu = np.triu_indices(len(spherical_correlation), k=1)

        spherical_correlation[spherical_tril] = \
            np.abs(spherical_correlation_estimate[spherical_tril])
        spherical_correlation[spherical_triu] = 1./2. + \
            np.angle(spherical_correlation_estimate[spherical_triu]) \
            / (2 * np.pi)

        plt.figure()
        sns.heatmap(
            spherical_correlation, vmin=0., vmax=1., cmap='Blues', square=True
        )
        if params.plot_estimate:
            plt.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))

        # Export spherical--Cartesian cross-correlation.
        hybrid_data = [
            list(
                spherical_map_data.vectorise(params.pivots[0]).tolist() +
                cartesian_map_data.vectorise(params.pivots[1]).tolist()
            )
            for spherical_map_data, cartesian_map_data
            in zip(spherical_data, cartesian_data)
        ]

        split_index = np.searchsorted(
            list(map(np.isrealobj, hybrid_data[-1])), True
        )

        sample_mean = np.average(hybrid_data, axis=0)

        sample_covar = np.linalg.multi_dot([
            np.conj(hybrid_data - sample_mean).T, hybrid_data - sample_mean
        ]) / (len(hybrid_data) - 1)

        sample_corr = covar_to_corr(sample_covar)

        output_file = output_dir/output_filename.format(
            'cross',
            "[{},{},{}]".format(params.kmin, params.khyb, params.kmax),
            order_tag
        )

        np.save(output_file, sample_corr)

        plt.figure()
        plt.subplot2grid((1, 2), (0, 0))
        sns.heatmap(
            np.real(sample_corr[split_index:, :split_index]),
            cmap='coolwarm', center=0
        )
        plt.subplot2grid((1, 2), (0, 1))
        sns.heatmap(
            np.real(sample_corr[split_index:, :split_index]),
            cmap='coolwarm', center=0
        )
        plt.subplots_adjust(wspace=0.25)
        if params.plot_estimate:
            plt.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))

if __name__ == '__main__':

    params = initialise_parameters()

    mask_tag, selection_tag, order_tag, pivot_tag = write_tags()

    try:
        source_range = sorted(params.map_source_names, key=int)
    except ValueError:
        source_range = sorted(params.map_source_names)

    # Set I/O paths.
    input_dir = data_dir/params.map_dir
    input_filename = "random-map-({})".format(",".join([
        "source={}", "map={}", "scale=[{},{}]", "orders={}",
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    output_dir = data_dir/"raw"/"survey_products"
    output_filename = "covar-estimate-({})".format(",".join([
        "source={}-{}".format(source_range[0], source_range[-1]),
        "map={}", "scale={}", "orders={}",
        "mask={}".format(mask_tag), "selection={}".format(selection_tag),
    ]))

    # Extract/export/visualise map statistics.
    spherical_data, cartesian_data = extract_map_data()
    export_map_statistics()
