"""Synthesise random catalogue maps for correlation estimation.

"""
import os
import re
import sys
from argparse import ArgumentParser

import numpy as np

try:
    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.mapper import RandomCatalogue, SphericalFKPCatalogue
    from harmonia.mapper import CartesianMap, SphericalMap
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )
    from harmonia.utils import Progress
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.mapper import RandomCatalogue, SphericalFKPCatalogue
    from harmonia.mapper import CartesianMap, SphericalMap
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )
    from harmonia.utils import Progress


@display_args(logger=logger)
def initialise():
    """Initialise the program parameters passed from ``stdin``.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed parameter namespace.

    """
    parser = ArgumentParser()

    parser.add_argument(
        '--iter', type=int, default=100,
        help="number of map samples to produce"
    )
    parser.add_argument('--batch', type=int, default=None, help="bacth number")
    parser.add_argument(
        '--only-cartesian', action='store_true',
        help="do not produce spherial maps"
    )

    parser.add_argument(
        '--kmin', type=float, default=None,
        help="minimum wavenumber of the map to make"
    )
    parser.add_argument(
        '--khyb', type=float, default=None,
        help="hybrid wavenumber of the map to make"
    )
    parser.add_argument(
        '--kmax', type=float, default=0.1,
        help="maximum wavenumber of the map to make"
    )
    parser.add_argument(
        '--orders', type=int, nargs='+', default=None,
        help="power spectrum multipoles of the Cartesian map to make"
    )

    parser.add_argument(
        '--density', type=float, default=1.e-4,
        help="mean number density of the primary random catalogue"
    )
    parser.add_argument(
        '--contrast', type=float, default=10.,
        help=(
            "this times higher mean number density "
            "for the secondary random catalogue"
        )
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

    parsed_args = parser.parse_args()

    return parsed_args


def tag():
    """Write output file tags.

    Returns
    -------
    mask_info, selection_info, order_info : str
        Mask, selection, multipole order information strings.

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

    return mask_info, selection_info, order_info


def manufacture():
    """Manufacture maps samples.

    """
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
    progress = Progress(
        params.iter, num_checkpts=5, process_name='synthesis', logger=logger
    )
    for iter_num, _ in enumerate(range(params.iter)):

        batch_no = (params.batch - 1) * params.iter + iter_num + 1  # 1-indexed

        catalogues = SphericalFKPCatalogue(
            radius,
            data_catalogue=RandomCatalogue(params.density, params.boxsize),
            contrast=params.contrast,
            mask=mask,
            selection=selection,
            random_seed=42,
            apply_selection_as_veto=apply_selection_as_veto
        )

        output_map = output_filename.format(
            batch_no, 'cartesian', params.khyb, params.kmax,
            str(params.orders).replace(" ", "")
        )

        cartesian_map = CartesianMap(
            catalogues, params.orders,
            kmin=params.khyb, kmax=params.kmax, num_mesh=params.mesh
        )
        cartesian_map_data = cartesian_map.power_multipoles
        cartesian_map_data.save(output_dir/output_map, 'npz')

        if not params.only_cartesian:
            disc = DiscreteSpectrum(radius, 'dirichlet', params.khyb)

            output_map = output_filename.format(
                batch_no, 'spherical', params.kmin, params.khyb, None
            )

            spherical_map = SphericalMap(catalogues, disc)
            spherical_map_data = spherical_map.density_contrast
            spherical_map_data.save(output_dir/output_map, 'npz')

        progress.report(iter_num)


if __name__ == '__main__':

    params = initialise()

    mask_tag, selection_tag, order_tag = tag()

    # Set I/O paths.
    mask_or_selection_dir = data_dir/"processed"/"survey_specifications"

    output_dir = data_dir/"raw"/"random_maps"
    output_filename = "random-map-({})".format(",".join([
        "source={}", "map={}", "scale=[{},{}]", "orders={}",
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    # Process catalogues.
    confirm_directory(output_dir)
    manufacture()
