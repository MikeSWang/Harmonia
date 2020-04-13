"""Produce the survey selection function using random catalogues.

"""
import os
import sys
from argparse import ArgumentParser

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

try:
    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.cosmology import BaseModel
    from harmonia.surveyor import generate_selection_samples, sky_to_spherical
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.cosmology import BaseModel
    from harmonia.surveyor import generate_selection_samples, sky_to_spherical


@display_args(logger=logger)
def initialise_parameters():
    """Initialise the program parameters passed from ``stdin``.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed parameter namespace.

    """
    parser = ArgumentParser(prog='produce selection function')

    parser.add_argument(
        '--source-dir', type=str, default="external/catalogues",
        help="catalogue source directory relative to 'storage/'"
    )
    parser.add_argument(
        '--source-file', type=str, required=True,
        help="catalogue source filename with extension"
    )
    parser.add_argument(
        '--cosmology-file', type=str, required=True,
        help=(
            "cosmological parameter file "
            "relative to 'storage/external/cosmology/'"
        )
    )

    parser.add_argument(
        '--coord', choices=['z', 'r'], required=True,
        help="coordinate of the sampled selection function"
    )
    parser.add_argument(
        '--max-coord', type=float, required=True,
        help=(
            "'coord_scale' parameter to pass to `generate_selection_samples`"
        )
    )

    parser.add_argument(
        '--nside', type=int, required=True,
        help=(
            "'NSIDE' parameter to pass to 'healpy' map; "
            "must be consistent with paired mask function sky map"
        )
    )
    parser.add_argument(
        '--bins', default=None,
        help=(
            "'num_bins' parameter to pass to `generate_selection_samples`"
        )
    )

    parser.add_argument(
        '--plot-distr', action='store_true',
        help="save plot of the selection for the specified coordinate"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def define_survey_selection(random_catalogue_file):
    """Define survey selection functionfrom a random catalogue.

    Parameters
    ----------
    random_catalogue_file : *str or* :class:`pathlib.Path`
        Random catalogue source file.

    Returns
    -------
    selection_function : float :class:`numpy.ndarray`
        Selection function samples at discrete radii.
    sky_fraction : float
        Sky fractional coverage.

    """
    # pylint: disable=no-member
    # Read catalogue Z, RA and DEC columns.
    with fits.open(random_catalogue_file) as random_catalogue_source:
        random_catalogue = random_catalogue_source[1].data
        z_samples = random_catalogue['Z']
        dec, ra = random_catalogue['DEC'], random_catalogue['RA']
    logger.info("Loaded random catalogue file %s.", random_catalogue_file)

    # Determine sky fraction.
    sky_mask = np.zeros(hp.nside2npix(params.nside))
    hit_coords = sky_to_spherical(np.column_stack((dec, ra)))
    hit_pixels = hp.ang2pix(params.nside, hit_coords[:, 0], hit_coords[:, 1])
    sky_mask[hit_pixels] = 1
    sky_fraction = np.sum(sky_mask) / np.size(sky_mask)

    # Load cosmology for z-to-r conversion.
    cosmo = BaseModel(cosmo_dir/params.cosmology_file)

    try:
        bins = int(params.bins)
    except (TypeError, ValueError):
        bins = params.bins

    sampled_selection = tuple(reversed(generate_selection_samples(
        params.coord, params.max_coord, z_samples,
        cosmo=cosmo, sky_fraction=sky_fraction, bins=bins
    )))

    logger.info("Selection function sampled.")

    return sampled_selection, sky_fraction


def export_survey_selection(sampled_selection, sky_fraction):
    """Export and visualise the survey selection function.

    Parameters
    ----------
    sampled_selection : tuple of float :class:`numpy.ndarray`
        Selection samples and corresponding coordinates.
    sky_fraction : float
        Sky fraction covered.

    """
    output_path = output_dir/output_filename.format(sky_fraction)

    # Write the sampled selection function out to a .txt file.
    output_file = output_path.with_suffix(output_path.suffix + '.txt')
    np.savetxt(
        output_file, np.column_stack(sampled_selection),
        fmt='%.9f', header='r, selection'
    )
    logger.info("Selection function samples written to %s.", output_path)

    # Visualise the pixelated map.
    output_file = output_path.with_suffix(output_path.suffix + '.pdf')
    unit = r' [Mpc/$h$]' if params.coord == 'r' else ''
    if params.plot_distr:
        fig = plt.figure("survey selection function")
        plt.plot(*sampled_selection)
        plt.xlabel(r'${}$'.format(params.coord) + unit)
        plt.ylabel(r'$\phi({})$'.format(params.coord))
        plt.title(r"{}, $f_\mathrm{{sky}}={:.2f}$".format(
            input_path.stem, sky_fraction
        ))
        fig.savefig(output_file)


if __name__ == '__main__':

    params = initialise_parameters()

    # Set I/O paths.
    cosmo_dir = data_dir/"cosmology"

    input_path = data_dir/params.source_dir/params.source_file

    output_dir = data_dir/"raw"/"survey_specifications"
    output_filename = "selection_func-({})".format(",".join([
        "source={}".format(params.source_file.split(".fits")[0]),
        "coord={}".format(params.coord),
        "nside={}".format(params.nside),
        "fsky={:.2f}"
    ]))

    confirm_directory(output_dir)

    # Define the survey selection and export/visualise selection
    # function samples.
    selection_func_samples, fsky = define_survey_selection(input_path)
    export_survey_selection(selection_func_samples, fsky)
