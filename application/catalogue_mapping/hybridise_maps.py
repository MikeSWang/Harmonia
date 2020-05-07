"""Hybridise spherical and Cartesian catalogue maps and summarise
statistics.

"""
import os
import re
import sys
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from application import application_logger as logger
    from application import data_dir, display_args
    from harmonia.algorithms import CartesianArray, SphericalArray
    from harmonia.utils import covar_to_corr
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import data_dir, display_args
    from harmonia.algorithms import CartesianArray, SphericalArray
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
        '--map-dir', type=str, default='raw/catalogue_maps',
        help="catalogue map directory relative to 'storage/'"
    )
    parser.add_argument(
        '--map-file-extension', type=str, nargs=2, default=['.npz', '.npz'],
        help="catalogue map file extension"
    )
    parser.add_argument(
        '--map-source-root', type=str, default='',
        help="catalogue source filename common root"
    )
    parser.add_argument(
        '--map-source-names', type=str, nargs='+', required=True,
        help="catalogue source filenames without extension"
    )

    parser.add_argument(
        '--pivots', type=str, nargs=2, default=['spectral', 'wavenumber'],
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
        help="power spectrum multipole orders of the Cartesian map"
    )

    parser.add_argument(
        '--rsd', action='store_true',
        help="velocity displacement added to create RSD effects"
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
        '--plot-correlation', action='store_true',
        help="plot correlation matrix of hybrid maps"
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
            "source=(?P<source>.*?),", params.mask_file
        ).group('source')
    else:
        mask_info = params.sky_fraction

    if params.selection_file is not None:
        # Only take selection '(source=<>,...)' information.
        selection_info = re.search(
            "source=(?P<source>.*?),", params.selection_file
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


def hybridise_maps():
    """Load and hybridise map data.

    Returns
    -------
    hybrid_map_data_array : list of :class:`numpy.ndarray`
        Hybrid map data array.

    """
    source_names = params.map_source_names.split() \
        if isinstance(params.map_source_names, str) \
        else params.map_source_names

    hybrid_map_data_array = []
    for source_name in source_names:
        spherical_map_path = input_dir/input_filename.format(
            source_name, 'spherical', params.kmin, params.khyb, None
        )
        cartesian_map_path = input_dir/input_filename.format(
            source_name, 'cartesian', params.khyb, params.kmax, order_tag
        )

        spherical_map_data = SphericalArray.load(
            spherical_map_path.with_suffix(
                spherical_map_path.suffix + params.map_file_extension[0]
            )
        )
        cartesian_map_data = CartesianArray.load(
            cartesian_map_path.with_suffix(
                cartesian_map_path.suffix + params.map_file_extension[1]
            )
        )

        hybrid_map_data_array.append(list(
            spherical_map_data.vectorise(params.pivots[0]).tolist()
            + cartesian_map_data.vectorise(params.pivots[1]).tolist()
        ))

    return hybrid_map_data_array


def export_hybrid_maps(hybrid_map_data, thredshold=0.):
    """Export and visualise statistical properties of hybrid map data.

    Parameters
    ----------
    hybrid_map_data : list of :class:`numpy.ndarray`
        List of hybrid map data vectors.
    thredshold : float, optional
        Absolute value threshold for plotting correlation matrix elements.

    """
    output_file = output_dir/output_filename.replace("hybrid", "hybrid-map")

    np.save(output_file, hybrid_map_data)

    # Find the split index between spherical and Cartesian parts/
    split_index = np.searchsorted(
        list(map(np.isrealobj, hybrid_map_data[-1])), True
    )

    # Find sample mean and correlation.
    sample_mean = np.average(hybrid_map_data, axis=0)

    if len(hybrid_map_data) > 1:
        sample_cov = np.linalg.multi_dot([
            np.conj(hybrid_map_data - sample_mean).T,
            hybrid_map_data - sample_mean
        ]) / (len(hybrid_map_data) - 1)
        sample_corr = covar_to_corr(sample_cov)
    else:
        warnings.warn(
            "Insufficient number of maps for estimating "
            "hybrid map correlation matrix."
        )
        return

    # Get modulus (lower triangular) and phase (upper triangular) values from
    # the spherical diagonal block.
    spherical_correlation = sample_corr[:split_index, :split_index]

    spherical_tril = np.tril_indices(len(spherical_correlation))
    spherical_triu = np.triu_indices(len(spherical_correlation), k=1)

    spherical_correlation[spherical_tril] = \
        np.abs(spherical_correlation[spherical_tril])
    spherical_correlation[spherical_triu] = 1./2. + \
        np.angle(spherical_correlation[spherical_triu]) / (2 * np.pi)

    # Apply threshold to lower triangular spherical part
    # to enhance visualisation.
    spherical_lower_tri = spherical_correlation[spherical_tril]
    spherical_lower_tri[np.abs(spherical_lower_tri) < thredshold] = 0.
    spherical_correlation[spherical_tril] = spherical_lower_tri

    # Get Cartesian correlation matrix block and
    # cross-correlation off-diagonal block.
    cartesian_correlation = sample_corr[split_index:, split_index:]

    cross_correlation = sample_corr[split_index:, :split_index]

    sns.set(style='ticks', font='serif')
    plt.figure("hybrid map statistics", figsize=(20, 3.5))

    plt.subplot2grid((1, 4), (0, 0))
    sns.heatmap(
        np.abs(spherical_correlation), cmap='Blues',
        square=True, vmin=0., vmax=1.
    )
    plt.subplot2grid((1, 4), (0, 1))
    sns.heatmap(
        np.real_if_close(cartesian_correlation, tol=10*10), cmap='coolwarm',
        square=True, center=0., vmin=-1., vmax=1.
    )
    plt.subplot2grid((1, 4), (0, 2))
    sns.heatmap(
        np.real(cross_correlation), cmap='coolwarm',
        center=0., vmin=-1., vmax=1.
    )
    plt.subplot2grid((1, 4), (0, 3))
    sns.heatmap(
        np.imag(cross_correlation), cmap='coolwarm',
        center=0., vmin=-1., vmax=1.
    )

    plt.subplots_adjust(wspace=0.25)

    output_path = output_dir/output_filename.replace("hybrid", "hybrid-corr")
    if params.plot_correlation:
        plt.savefig(output_path.with_suffix(output_path.suffix + '.pdf'))


if __name__ == '__main__':

    params = initialise_parameters()

    mask_tag, selection_tag, order_tag, pivot_tag = write_tags()

    # Set I/O paths.
    input_dir = data_dir/params.map_dir
    input_filename = "catalogue-map-({})".format(",".join([
        "source={}", "map={}", "scale=[{},{}]", "orders={}",
        "rsd={}".format(params.rsd),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    output_dir = input_dir
    output_filename = "hybrid-({})".format(",".join([
        "source={}".format(params.map_source_root),
        "scale=[{},{},{}]".format(params.kmin, params.khyb, params.kmax),
        "orders={}".format(order_tag),
        "rsd={}".format(params.rsd),
        "pivots={}".format(pivot_tag),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    # Hybridise maps and export/visualise data statistics.
    hybrid_maps = hybridise_maps()
    export_hybrid_maps(hybrid_maps)
