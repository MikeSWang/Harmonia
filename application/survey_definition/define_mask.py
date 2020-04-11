"""Produce the survey mask map using random catalogues.

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
    from harmonia.surveyor import sky_to_spherical
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))

    from application import application_logger as logger
    from application import confirm_directory, data_dir, display_args
    from harmonia.surveyor import sky_to_spherical


@display_args(logger=logger)
def initialise():
    """Initialise the program parameters passed from ``stdin``.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed parameter namespace.

    """
    parser = ArgumentParser(prog='produce mask map')

    parser.add_argument(
        '--source-dir', type=str, default="external/catalogues",
        help="catalogue source directory relative to 'storage/'"
    )
    parser.add_argument(
        '--source-file', type=str, required=True,
        help="catalogue source filename with extension"
    )

    parser.add_argument(
        '--nside', type=int, default=512,
        help="'NSIDE' parameter to pass to 'healpy' map"
    )

    parser.add_argument(
        '--overwrite-map', action='store_true',
        help="overwrite any existing 'healpy' map with the same output file"
    )
    parser.add_argument(
        '--plot-map', action='store_true',
        help="save plot of the 'healpy' mask map"
    )

    parsed_args = parser.parse_args()

    return parsed_args


def define(random_catalogue_file):
    """Process a random catalogue file into a `healpy` sky map.

    Parameters
    ----------
    random_catalogue_file : *str or* :class:`pathlib.Path`
        Random catalogue source file.

    Returns
    -------
    sky_map : float :class:`numpy.ndarray`
        Pixelated sky map for a veto mask.

    """
    # Read catalogue RA and DEC columns.
    with fits.open(random_catalogue_file) as random_catalogue_source:
        # pylint: disable=no-member
        random_catalogue = random_catalogue_source[1].data
        dec, ra = random_catalogue['DEC'], random_catalogue['RA']
    logger.info("Loaded random catalogue file %s.", random_catalogue_file)

    # Transform coordinates and determine pixel map resolution.
    hit_coords = sky_to_spherical(np.column_stack((dec, ra)))

    resolution = hp.nside2resol(params.nside, arcmin=True) / 60
    logger.info("Sky map resolution is %.2f degrees.", resolution)

    # Pixelate catalogue map.
    sky_map = np.zeros(hp.nside2npix(params.nside))
    hit_pixels = hp.ang2pix(params.nside, hit_coords[:, 0], hit_coords[:, 1])
    sky_map[hit_pixels] = 1
    logger.info("Sky map pixelation of input catalogue finished.")

    return sky_map


def export_mask(sky_map):
    """Export and visualise a pixelated sky mask map.

    Parameters
    ----------
    sky_map : :class:`numpy.ndarray`

    """
    output_path = output_dir/output_filename

    # Write the pixelated map out to a .fits file.
    output_file = output_path.with_suffix(output_path.suffix + '.fits')
    hp.write_map(
        output_file, sky_map, dtype=float, overwrite=params.overwrite_map
    )
    logger.info("Pixelated map written to %s.", output_path)

    # Visualise the pixelated map.
    output_file = output_path.with_suffix(output_path.suffix + '.pdf')
    if params.plot_map:
        num = 0
        fig = plt.figure(num)
        hp.mollview(
            sky_map, fig=num, cmap='binary_r', cbar=False,
            title=r"{}, $n_\mathrm{{side}}={}$".format(
                input_path.stem, params.nside
            )
        )
        fig.savefig(output_file)


if __name__ == '__main__':

    params = initialise()

    # Set I/O paths.
    input_path = data_dir/params.source_dir/params.source_file

    output_dir = data_dir/"raw"/"survey_specifications"
    output_filename = "mask_map-({})".format(",".join([
        "source={}".format(params.source_file.split(".fits")[0]),
        "nside={}".format(params.nside)
    ]))

    # Process the random catalogue into a mask map and export/visualise.
    confirm_directory(output_dir)
    mask_map = define(input_path)
    export_mask(mask_map)
