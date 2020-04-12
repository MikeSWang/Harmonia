"""Transform catalogues into hybrid maps.

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
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.mapper import SphericalFKPCatalogue
    from harmonia.mapper import CartesianMap, SphericalMap
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.mapper import SphericalFKPCatalogue
    from harmonia.mapper import CartesianMap, SphericalMap
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )

CATALOGUE_HEADER = ["x", "y", "z", "vx", "vy", "vz", "mass"]


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
        '--map', type=str, choices=['spherical', 'cartesian'],
        help="coordinate system of the map to make"
    )
    parser.add_argument(
        '--kmin', type=float, default=None,
        help="minimum wavenumber of the map to make"
    )
    parser.add_argument(
        '--kmax', type=float, default=0.1,
        help="maximum wavenumber of the map to make"
    )
    parser.add_argument(
        '--orders', type=int, nargs='+', default=None,
        help="power spectrum multipole orders of the Cartesian map to make"
    )

    parser.add_argument(
        '--source-dir', type=str, default='external/simulations',
        help="catalogue source directory relative to 'storage/'"
    )
    parser.add_argument(
        '--source-file', type=str, required=True,
        help="catalogue source filename with extension"
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
        '--boxsize', type=float, default=1000.,
        help="catalogue box size per dimension"
    )
    parser.add_argument(
        '--mesh', type=int, default=256,
        help="mesh number per dimension for Cartesian map grid assignment"
    )
    parser.add_argument(
        '--rsd', action='store_true',
        help="add velocity displacement to create RSD effects"
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
        '--plot-catalogue', action='store_true',
        help="plot processed catalogue"
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


def transform_catalogues():
    """Transform input catalogues to maps.

    Returns
    -------
    |CartesianMap| or |SphericalMap|
        Catalogue map.


    .. |CartesianMap| replace::

        :class:`harmonia.mapper.map_transform.CartesianMap`


    .. |SphericalMap| replace::

        :class:`harmonia.mapper.map_transform.SphericalMap`

    """
    output_path = output_dir/output_filename

    radius = params.boxsize / 2.

    # Apply any survey mask or selection functions.
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

    # Load and make catalogues.
    catalogues = SphericalFKPCatalogue(
        radius,
        source_file=input_path,
        source_kwargs={
            'headings': CATALOGUE_HEADER,
            'mean_density': params.density,
            'boxsize': params.boxsize,
            'offset': params.rsd,
        },
        contrast=params.contrast,
        mask=mask,
        selection=selection,
        random_seed=42,
        apply_selection_as_veto=apply_selection_as_veto
    )

    # Transform catalogues to maps.
    if params.map == 'spherical':

        disc = DiscreteSpectrum(radius, 'dirichlet', params.kmax)
        spherical_map = SphericalMap(catalogues, disc)
        spherical_map_data = spherical_map.density_contrast
        spherical_map_data.save(output_path, 'npz')

        return spherical_map

    if params.map == 'cartesian':
        cartesian_map = CartesianMap(
            catalogues, params.orders,
            kmin=params.kmin, kmax=params.kmax, num_mesh=params.mesh
        )
        cartesian_map_data = cartesian_map.power_multipoles
        cartesian_map_data.save(output_path, 'npz')

        return cartesian_map

    raise ValueError(f"Unknown map catagory: {params.map}.")


def view_catalogue_map():
    """Visualise processed catalogues.

    """
    boxsize = catalogue_map.catalogues.data_catalogue.attrs['BoxSize']

    extent = np.mean(boxsize) / 2

    data_mesh = catalogue_map.catalogues.data_catalogue.to_mesh(
        Nmesh=params.mesh, resampler='tsc', compensated=True
    )
    rand_mesh = catalogue_map.catalogues.random_catalogue.to_mesh(
        Nmesh=params.mesh, resampler='tsc', compensated=True
    )

    plt.figure("painted catalogues", figsize=(11, 6))
    for idx, axes in enumerate([[0, 1], [0, 2], [1, 2]]):
        plt.subplot2grid((2, 3), (0, idx))
        plt.imshow(
            data_mesh.preview(axes=axes), cmap='coolwarm',
            extent=[- extent, extent, - extent, extent]
        )
        plt.subplot2grid((2, 3), (1, idx))
        plt.imshow(
            rand_mesh.preview(axes=axes), cmap='coolwarm',
            extent=[- extent, extent, - extent, extent]
        )

    output_path = output_dir/output_filename

    plt.subplots_adjust(hspace=0.2, wspace=0.25)
    plt.savefig(output_path.with_suffix(output_path.suffix + '.pdf'))


if __name__ == '__main__':

    params = initialise_parameters()

    mask_tag, selection_tag, order_tag = write_tags()

    # Set I/O paths.
    mask_or_selection_dir = data_dir/"processed"/"survey_specifications"

    input_path = data_dir/params.source_dir/params.source_file

    output_dir = data_dir/"raw"/"catalogue_maps"
    output_filename = "catalogue-map-({})".format(",".join([
        "source={}".format(params.source_file.rstrip(".txt")),
        "map={}".format(params.map),
        "scale=[{},{}]".format(params.kmin, params.kmax),
        "orders={}".format(order_tag),
        "rsd={}".format(params.rsd),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    confirm_directory(output_dir)

    # Transform catalogues and save maps/visualise processed catalogues.
    catalogue_map = transform_catalogues()
    if params.plot_catalogue:
        view_catalogue_map()
