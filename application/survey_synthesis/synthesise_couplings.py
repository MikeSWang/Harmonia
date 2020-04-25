"""Pre-compute spherical coupling coefficients for baseline cosmology
and survey specifications.

"""
import os
import re
import sys
from argparse import ArgumentParser

import numpy as np
from mpi4py import MPI

try:
    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.algorithms import DiscreteSpectrum
    from harmonia.reader import Couplings
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
    from harmonia.reader import Couplings
    from harmonia.surveyor import (
        generate_mask_by_sky_fraction,
        generate_mask_from_map,
        generate_selection_by_cut,
        generate_selection_by_distribution,
        generate_selection_from_samples,
    )

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
        '--redshift', type=float, default=1.,
        help="redshift for model evaluation"
    )
    parser.add_argument(
        '--kmax', type=float, default=None,
        help="maximum wavenumber of the spherical Fourier spectrum"
    )
    parser.add_argument(
        '--rmax', type=float, default=500.,
        help="maximum radius at which boundary conditions are applied"
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
        '--pixelate', type=int, default=None,
        help='pixelate the mask map/function for integration'
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

    return parser.parse_args()


def write_tags():
    """Write output file tags.

    Returns
    -------
    mask_info, selection_info : str
        Mask and selection information strings.

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

    return mask_info, selection_info


def synthesise_couplings():
    """Synthesise spherical couplings from baseline cosmology and
    survey specifications.

    Returns
    -------
    :class:`harmonia.reader.couplings.Couplings`
        Spherical couplings.

    """
    # Apply any survey mask or selection functions.
    nside = params.pixelate

    if params.mask_file is not None:
        mask = generate_mask_from_map(
            'spherical', mask_map_file=mask_or_selection_dir/params.mask_file
        )
    elif params.sky_fraction is not None:
        mask = generate_mask_by_sky_fraction(
            'spherical', sky_fraction=params.sky_fraction
        )
    else:
        mask = None

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
            params.rmax, distribution, **distr_kwargs
        )
    elif params.selection_cut is not None:
        selection = generate_selection_by_cut(*params.selection_cut)
    else:
        selection = None

    disc = DiscreteSpectrum(params.rmax, 'dirichlet', params.kmax, comm=comm)
    _couplings = Couplings(
        disc, survey_specs=dict(mask=mask, selection=selection),
        pixelate=nside, comm=comm
    )

    return _couplings


if __name__ == '__main__':

    params = initialise_parameters()

    mask_tag, selection_tag = write_tags()

    # Set I/O paths.
    mask_or_selection_dir = data_dir/"processed"/"survey_specifications"

    output_dir = data_dir/"raw"/"survey_products"
    output_filename = "couplings-({})".format(",".join([
        "kmax={}".format(params.kmax),
        "mask={}".format(mask_tag),
        "selection={}".format(selection_tag),
    ]))

    confirm_directory(output_dir)

    # Compute and save couplings.
    couplings = synthesise_couplings()
    couplings.save(output_dir/output_filename)
