"""Determine the survey window/mask multipoles for Cartesian two-point
correlators.

"""
import os
import re
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

try:
    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )
    from harmonia.surveyor.synthesis import SyntheticCatalogue
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )
    from harmonia.surveyor.synthesis import SyntheticCatalogue


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
        '--orders', type=int, nargs='+', default=[0, 2, 4, 6, 8],
        help="multipole orders of the window/mask function"
    )

    parser.add_argument(
        '--density', type=float, default=5.e-3,
        help="mean number density of the synthetic catalogue"
    )
    parser.add_argument(
        '--boxsize', type=float, default=1000.,
        help="synthetic catalogue box size per dimension"
    )
    parser.add_argument(
        '--expansion', type=float, default=70.,
        help="expansion of boxsize for measuring the window multipoles"
    )
    parser.add_argument(
        '--mesh', type=int, default=768,
        help="mesh number per dimension for measuring the window multipoles"
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
        '--plot-multipoles', action='store_true',
        help="plot the determined window/mask function"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def write_tags():
    """Write output file tags.

    Returns
    -------
    mask_info, selection_info, order_info : str
        Mask, selection, multipole order information strings.

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

    return mask_info, selection_info, order_info


def synthesise_window_and_mask():
    """Synthesise the window and mask multipoles from a high-density
    random catalogue.

    Returns
    -------
    |CartesianArray| and :class:`numpy.ndarray`
        Window and mask multipoles.


    .. |CartesianArray| replace::

        :class:`harmonia.algorithms.arrays.CartesianArray`

    """
    radius = params.boxsize / 2.

    if params.mask_file is not None:
        mask = generate_mask_from_map(
            'cartesian', mask_map_file=mask_or_selection_dir/params.mask_file
        )
    elif params.sky_fraction is not None:
        mask = generate_mask_by_sky_fraction(
            'cartesian', sky_fraction=params.sky_fraction
        )
    else:
        mask = None

    apply_selection_as_veto = False
    if params.selection_file is not None:
        selection = generate_selection_from_samples(
            *np.loadtxt(
                mask_or_selection_dir/params.selection_file, unpack=True
            )
        )
    elif params.selection_distribution is not None:
        distribution = params.selection_distribution[0]
        if distribution == 'gaussian':
            distr_kwargs = dict(zip(
                ['peak', 'scale', 'location'],
                list(map(float, params.selection_distribution[1:]))
            ))
        elif distribution == 'gamma':
            distr_kwargs = dict(zip(
                ['peak', 'scale', 'shape'],
                list(map(float, params.selection_distribution[1:]))
            ))
        else:
            raise ValueError("Unsupported distribution for selection.")
        selection = generate_selection_by_distribution(
            radius, distribution, **distr_kwargs
        )
    elif params.selection_cut is not None:
        apply_selection_as_veto = True
        selection = generate_selection_by_cut(*params.selection_cut)
    else:
        selection = None

    synthetic_catalogue = SyntheticCatalogue(
        params.density, params.boxsize,
        expansion=params.expansion,
        sphericalise=radius,
        mask=mask, selection=selection,
        apply_selection_as_veto=apply_selection_as_veto
    )

    window_function = synthetic_catalogue.compute_window(
        params.orders, num_mesh=params.mesh
    )

    mask_function = synthetic_catalogue.compute_mask(params.orders)

    return window_function, mask_function


def export_window_and_mask():
    """Export and visualise window/mask multipoles from the
    synthetic catalogue.

    """
    output_window_file = \
        output_dir/output_filename.replace("mask_or_window", "window")

    window_multipoles.save(output_window_file, '.npz')

    output_mask_file = \
        output_dir/output_filename.replace("mask_or_window", "mask")

    np.save(output_mask_file, mask_multipoles)

    plt.figure("window/mask function multipoles", figsize=(8, 11))

    plt.subplot2grid((2, 1), (0, 0))
    for ell in params.orders:
        plt.semilogx(
            window_multipoles.attrs['wavenumbers'],
            window_multipoles.array['power'][
                window_multipoles.array['order'] == ell
            ],
            label=r'$\ell={}$'.format(ell)
        )
    plt.ylim(-1.1, 1.1)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$Q_\ell(k)$")
    plt.legend()

    plt.subplot2grid((2, 1), (1, 0))
    for ell in params.orders:
        plt.semilogx(
            mask_multipoles['separation'][mask_multipoles['order'] == ell],
            mask_multipoles['correlation'][mask_multipoles['order'] == ell],
            label=r'$\ell={}$'.format(ell)
        )
    plt.ylim(-1.1, 1.1)
    plt.xlabel(r"$s$")
    plt.ylabel(r"$Q_\ell(s)$")
    plt.legend()

    if params.plot_multipoles:
        outplot = output_dir/output_filename
        plt.savefig(outplot.with_suffix(outplot.suffix + '.pdf'))


if __name__ == '__main__':

    params = initialise_parameters()

    mask_tag, selection_tag, order_tag = write_tags()

    # Set I/O paths.
    mask_or_selection_dir = data_dir/"processed"/"survey_specifications"

    output_dir = data_dir/"raw"/"survey_products"
    output_filename = "mask_or_window-({})".format(",".join([
        "orders={}".format(order_tag),
        "boxsize={}".format(params.boxsize),
        "expansion={}".format(params.expansion),
        "mesh={}".format(params.mesh),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    confirm_directory(output_dir)

    # Synthesise survey window/mask multipoles and export/visualise multipoles.
    window_multipoles, mask_multipoles = synthesise_window_and_mask()
    export_window_and_mask()
