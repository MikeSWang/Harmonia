"""
Definition (:mod:`~harmonia.surveyor.definition`)
===========================================================================

Produce survey definition and specifications for processing and analysing
cosmological data.

.. autosummary::

    generate_mask_by_sky_fraction
    generate_mask_from_map
    generate_selection_by_distribution
    generate_selection_from_samples
    generate_selection_samples
    generate_compression_matrix

|

"""
# pylint: disable=no-name-in-module
import healpy as hp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Interpolation
from scipy.special import loggamma
from scipy.stats import gamma, norm

from harmonia.reader.likelihoods import spherical_covariance
from harmonia.surveyor.coordinates import (
    sky_to_spherical,
    to_box_coords,
    _is_coord_3d,
)


def generate_mask_by_sky_fraction(coord_system, sky_fraction=1.,
                                  box_shift=None):
    """Generate mask function given the sky fraction corresponding to a
    spherical cap (rather than a wedge).

    Parameters
    ----------
    coord_system : {'cartesian', 'spherical', 'sky'}
        Coordinate system of the generated mask function.  If 'sky', the
        returned function can accept either 2-d or 3-d coordinates.
    sky_fraction : float, optional
        Sky coverage, ``0 < sky_fraction <= 1``.
    box_shift : float, array_like or None, optional
        Passed as `box_centre` to :func:`~.coordinates.to_box_coords` so
        that the resulting mask function accepts Cartesian coordinates
        with origin at a box corner.  Ignored unless `coord_system` is
        'cartesian'.

    Returns
    -------
    mask_function : callable
        A mask function accepting `coord_system` as its coordinate system
        and returning boolean values.

    """
    # Any conversion to the 'cartesian' coordinate system as internal
    # `mask_function` is defined for native spherical coordinates.
    if coord_system == 'cartesian':
        apply_to_box_coords_from_native = 'spherical'
    else:
        apply_to_box_coords_from_native = 'null'
        box_shift = None

    # `coords` can be either 'spherical' or 'sky' but the native
    # implementation is 'spherical'; the decorator takes care of
    # 'cartesian' inputs.
    @to_box_coords(apply_to_box_coords_from_native, box_centre=box_shift)
    def mask_function(coords):

        # Transform to spherical surface coords.
        surf_coords = sky_to_spherical(coords) if coord_system == 'sky' \
            else np.atleast_2d(coords)

        return surf_coords[:, -2] <= np.arccos(1 - 2 * sky_fraction)

    return mask_function


def generate_mask_from_map(coord_system, mask_map=None, nside=None,
                           mask_map_file=None, box_shift=None):
    """Generate mask function from a veto mask map.

    Parameters
    ----------
    coord_system : {'cartesian', 'spherical', 'sky'}
        Coordinate system of the generated mask function.  If 'sky', the
        returned function can accept either 2-d or 3-d coordinates.
    mask_map : :class:`numpy.ndarray` or None, optional
        A veto array corresponding to pixels in a `healpy` mask map with
        parameter `nside` (default is `None`).  Ignored if
        a `healpy`-generated .fits file for the mask map is provided.
    nside : int or None, optional
        'nside' parameter of the `healpy` mask map (default is `None`).
        Ignored if  a `healpy`-generated .fits file for the
        mask map is provided.
    mask_map_file : *str or* :class:`pathlib.Path`
        `healpy`-generated .fits file for the mask map.  Must contain
        the 'NSIDE' parameter in its header.
    box_shift : float, array_like or None, optional
        Passed as `box_centre` to :func:`~.coordinates.to_box_coords` so
        that the resulting mask function accepts Cartesian coordinates
        with origin at a box corner.  Ignored unless `coord_system` is
        'cartesian'.

    Returns
    -------
    mask_function : callable
        A mask function accepting `coord_system` as its coordinate system
        and returning boolean values.

    """
    if mask_map_file is not None:
        mask_map, mask_map_header = hp.read_map(
            mask_map_file, dtype=float, h=True, verbose=False
        )
        nside = dict(mask_map_header)['NSIDE']

    # Any conversion to the 'cartesian' coordinate system as internal
    # `mask_function` is defined for native spherical coordinates.
    if coord_system == 'cartesian':
        apply_to_box_coords_from_native = 'spherical'
    else:
        apply_to_box_coords_from_native = 'null'
        box_shift = None

    # `coords` can be either 'spherical' or 'sky' but the native
    # implementation is 'spherical'; the decorator takes care of
    # 'cartesian' inputs.
    @to_box_coords(apply_to_box_coords_from_native, box_centre=box_shift)
    def mask_function(coords):

        # Transform to spherical surface coords.
        surf_coords = sky_to_spherical(coords) if coord_system == 'sky' \
            else np.atleast_2d(coords)

        pixel = hp.ang2pix(nside, *surf_coords[:, [-2, -1]].T)

        return mask_map[pixel].astype(bool)

    return mask_function


def generate_selection_by_distribution(coord_scale, distribution, peak,
                                       scale=None, location=None, shape=None):
    """Generate selection function based on a probability distribution
    suitably rescale horizontally and vectically.

    Notes
    -----
    If the generated selection function detects 3-d input coordinate
    arrays, it would assume the coordinates are the Cartesian; otherwise
    it assumes the coordinates are the radial coordinate.

    Parameters
    ----------
    coord_scale : float
        Coordinate scale of the selection function, e.g. the maximum
        comoving radius of a catalogue.
    peak : float
        Peak value of the selection function used for renormalising
        the probability density function by its maximum value.
    distribution : {'gaussian', 'gamma'}
        Distribution to use, either 'gaussian' or 'gamma'.
    scale, location, shape : float or None, optional
        Scale, location and shape parameter of the distribution.  For
        a normal distribution, `scale` is the standard deviation and
        `location` is the mean in units of `coord_scale`; for a gamma
        distribution, see :class:`scipy.stats.gamma`.

    Returns
    -------
    callable
        Normalised selection function of the same coordinate as the sample
        coordinates.

    """
    def rescale_coords(coords):

        coords = np.linalg.norm(coords, axis=-1) if _is_coord_3d(coords) \
                else np.squeeze(coords)

        return coords / coord_scale

    def gaussian_selection(coords):

        r = rescale_coords(coords)

        pdf_peak = 1 / (np.sqrt(2 * np.pi) * scale)

        return peak / pdf_peak * norm.pdf(r, loc=location, scale=scale)

    def gamma_selection(coords):

        r = rescale_coords(coords) / (1 + 5 / np.sqrt(shape))

        pdf_peak = np.exp(
            (shape - 1) * np.log(shape - 1)
            - (shape - 1) - np.log(scale) - loggamma(shape)
        )

        return peak / pdf_peak * gamma.pdf(r, a=shape, scale=scale)

    if distribution == 'gaussian':
        return gaussian_selection

    if distribution == 'gamma':
        return gamma_selection

    raise ValueError(f"Unsupproted distribution: {distribution}.")


def generate_selection_by_cut(low_end, high_end):
    """Generate selection function at a constant value by a cut.

    Notes
    -----
    If the generated selection function detects 3-d input coordinate
    arrays, it would assume the coordinates are the Cartesian; otherwise
    it assumes the coordinates are the radial coordinate.

    Parameters
    ----------
    low_end, high_end : float
        Low and high end of the cut in some selection coordinate.

    Returns
    -------
    callable
        Constant cut selection function.

    """
    def selection_function(coords):

        coord = np.linalg.norm(coords, axis=-1) if _is_coord_3d(coords) \
            else np.squeeze(coords)

        return np.logical_and(
            np.greater_equal(coord, low_end), np.less_equal(coord, high_end)
        )

    return selection_function


def generate_selection_samples(selection_coordinate, coord_scale,
                               redshift_samples, cosmo, sky_fraction=1.,
                               bins=None):
    """Generate selection function samples from redshift samples.

    The selection function is normalised by the mean number density
    inferred from the samples and the sky fraction.

    Parameters
    ----------
    selection_coordinate : {'z', 'r'}
        Coordinate of the generated selection function, either redshift
        'z' or comoving distance 'r'.
    coord_scale : float
        Coordinate scale used to rescale the selection coordinates, e.g.
        the maximum comoving radius of the catalogue to which the
        selection function is applied.
    redshift_samples : float, array_like
        Redshift samples.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology`
        Cosmological model providing the redshift-to-distance conversion
        and thus normalisation of the selection function by volume.
    sky_fraction : float, optional
        The sky coverage of redshift samples as a fraction used to
        normalise the selection function (default is 1.),
        ``0 < sky_fraction <= 1``.
    bins : int, str or None, optional
        Number of bins or binning scheme used to generate the selection
        function samples (see :func:`numpy.histogram_bin_edges`).
        If `None` (default), Scott's rule for binning is used.

    Returns
    -------
    samples, coords : :class:`numpy.ndarray`
        Normalised (dimensionless) selection function samples and
        corresponding coordinates.

    """
    if not 0 < sky_fraction <= 1:
        raise ValueError("Sky fraction must be between 0 and 1.")

    # Perform binning with specified or optimal Scott's binning.
    redshift_samples = redshift_samples[redshift_samples >= 0.]

    bins = bins or 'scott'

    bin_edges = np.histogram_bin_edges(redshift_samples, bins=bins)

    bin_counts, z_edges = np.histogram(redshift_samples, bins=bin_edges)

    z_centres = (z_edges[:-1] + z_edges[1:]) / 2.

    # Find mean number density in each redshift shell bin and normalise
    # by overall density.
    r_edges = cosmo.comoving_distance(z_edges)

    bin_volumes = sky_fraction * (4./3. * np.pi) \
        * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    overall_nbar = np.sum(bin_counts) / np.sum(bin_volumes)

    bin_selections = (bin_counts / bin_volumes) / overall_nbar

    bin_centres = z_centres if selection_coordinate == 'z' \
        else cosmo.comoving_distance(z_centres)

    # Generate a selection function of the appropriate coordinate.
    interpolated_selection = Interpolation(bin_centres, bin_selections, ext=1)

    # Attempt resampling and reject volatile ends with negative
    # interpolated values.
    interpolated_selection_samples = interpolated_selection(
        np.linspace(bin_centres.min(), bin_centres.max(), num=len(bin_centres))
    )

    max_point = np.argmax(bin_counts)

    low_partition = interpolated_selection_samples[:max_point]
    high_partition = interpolated_selection_samples[max_point:]

    low_end = np.argmax(low_partition[::-1] < 0)
    high_end = np.argmax(high_partition < 0)

    burnt_remains = slice(max_point - low_end, max_point + high_end)

    coords = bin_centres[burnt_remains] * coord_scale / np.max(bin_centres)
    samples = bin_selections[burnt_remains]

    return samples, coords


def generate_selection_from_samples(sample_selections, sample_coords):
    """Generate selection function interpolated from normalised selection
    function samples at sample coodinates.

    Notes
    -----
    If the generated selection function detects 3-d input coordinate
    arrays, it would assume the coordinates are the Cartesian; otherwise
    it assumes the coordinates are the radial coordinate.

    Parameters
    ----------
    sample_selections, sample_coords : float :class:`numpy.ndarray`
        Selection function samples and coordinates.

    Returns
    -------
    callable
        Normalised selection function of the same coordinate as the sample
        coordinates.

    """
    def selection_function(coords):

        r = np.linalg.norm(coords, axis=-1) if _is_coord_3d(coords) \
            else np.squeeze(coords)

        return Interpolation(sample_coords, sample_selections, ext=1)(r)

    return selection_function


def generate_compression_matrix(fiducial_model_kwargs,
                                extremal_model_kwargs=None,
                                sensitivity_threshold=0.01, discard=None):
    r"""Generate a compression matrix for spherical modes.

    Notes
    -----
    Compression is achieved by discarding non-positive eigenvalue modes
    that are at least :math:`10^{-8}` times smaller than the largest and
    in addition any of the following means:

        * `discard` is passed to discard a number of low-eigenvalue modes;
        * `extremal_model_kwargs` is passed and eigenvalues of the
          resulting model covariance are compared with those from
          `fiducial_covariance`.  Modes corresponding to low, insensitive
          (i.e. relative difference less than `sensitivity_threshold`)
          are discarded.
        * A combination of the above if the appropriate parameters
          are passed.

    Parameters
    ----------
    fiducial_model_kwargs : dict
        Fiducial model parameters to be passed to
        :func:`~.reader.likelihoods.spherical_covariance`.
    extremal_model_kwargs : dict or None, optional
        Extremal model parameters to be passed to
        :func:`~.reader.likelihoods.spherical_covariance`.
    sensitivity_threshold: float, optional
        Sensitivity threshold for modes deemed discardable
        (default is 0.01).
    discard : int or None, optional
        Number of low-eigenvalue modes to discard from all modes
        (default is `None`).

    Returns
    -------
    compression_mat : :class:`numpy.ndarray`
        Compression matrix.

    """
    fiducial_covariance = spherical_covariance(**fiducial_model_kwargs)

    evals_fiducial, evecs = np.linalg.eigh(fiducial_covariance)

    selectors = []

    # Compression by positive magnitude.
    selectors.append(evals_fiducial > 1.e-8 * np.max(evals_fiducial))

    # Compression by discard.
    if discard is not None:
        selectors.append(np.indices(evals_fiducial) >= discard)

    # Compression by comparison for sensitivity.
    extremal_covariance = spherical_covariance(**extremal_model_kwargs)

    evals_extremal = np.linalg.eigvalsh(extremal_covariance)

    selectors.append(
        ~np.isclose(evals_extremal, evals_fiducial, rtol=sensitivity_threshold)
    )

    # pylint: disable=no-member
    # Compress and reverse order.
    evecs = evecs[np.logical_and.reduce(selectors)][:, ::-1]

    compression_mat = np.conj(evecs).T

    return compression_mat
