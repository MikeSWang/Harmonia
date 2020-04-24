"""
Catalogue maker (:mod:`~harmonia.mapper.catalogue_maker`)
===========================================================================

Make discrete catalogues from observed or simulated realisations.

.. autosummary::

    SourceCatalogue
    RandomCatalogue
    SphericalFKPCatalogue
    spherical_indicator

|

"""
import logging
import warnings

import numpy as np
from nbodykit.lab import CSVCatalog, FKPCatalog, UniformCatalog


def spherical_indicator(cartesian_position, bounding_radius):
    """Indicate whether an object lies within the a spherical domain
    centred at the origin.

    Parameters
    ----------
    cartesian_position : float, array_like
        Object position in Cartesian coordinates.
    bounding_radius : float
        Bounding radius of the spherical domain.

    Returns
    -------
    indication : bool :class:`numpy.ndarray`
        `True` if the objection position lies within the spherical domain.

    """
    indication = (
        np.linalg.norm(cartesian_position, axis=-1) <= bounding_radius
    )

    return indication


class RandomCatalogue(UniformCatalog):
    """Uniform random catalogue.

    Notes
    -----
    Origin of the catalogue is at a catalogue box corner.

    Parameters
    ----------
    mean_density : float
        Desired mean particle number density (in cubic :math:`h`/Mpc).
    boxsize : float, array_like
        Catalogue boxsize (in Mpc/:math:`h`) as a scalar or a triple of
        scalars.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).

    """

    def __init__(self, mean_density, boxsize, seed=None, comm=None):

        super().__init__(mean_density, boxsize, seed=seed, comm=comm)

        self.attrs['BoxSize'] = [boxsize,] * 3 \
            if np.isscalar(boxsize) else boxsize
        self.attrs['nbar'] = mean_density

        if self.comm is None or self.comm.rank == 0:
            self.logger.debug(
                "%s generated with origin at the box centre.", self
            )

    def __str__(self):

        str_info = "nbar={}, boxsize={}, seed={}".format(
            self.attrs['nbar'], self.attrs['BoxSize'], self.attrs['seed']
        )

        return f"{self.__class__.__name__}({str_info})"


class SourceCatalogue(CSVCatalog):
    """Catalogue from external sources.

    Notes
    -----
    This assumes the external catalogue has a Cartesian coordinate system
    with the origin at a corner of the Cartesian box.

    Parameters
    ----------
    source_file : str
        Catalogue source file.
    headings : list of str
        Column headings of the catalogue file.  This must contain three
        columns 'x', 'y' and 'z' for holding the Cartesian coordinates.
    mean_density: float or None, optional
        If not `None` (default), this overrides default catalogue mean
        number density (in cubic :math:`h`/Mpc) (e.g. to evade integral
        constraint).
    boxsize : float or None, optional
        If not `None` (default), this overrides default catalogue boxsize
        (in Mpc/:math:`h`) in case of unit upscaling.
    upscale : float, optional
        Scaling factor for converting any length unit to Mpc/:math:`h`
        (default is 1.), e.g. ``upscale=1.e-3`` for converting
        Kpc/:math:`h` to Mpc/:math:`h`.
    offset : bool, optional
        If `True` (default is `False`), add the velocity columns to
        position columns (for e.g. redshift-space distortions).
    offset_upscale : float, optional
        Scaling factor for converting the velocity offset length unit to
        Mpc/:math:`h` (default is 1.e-3), e.g. ``offset_upscale=1.e-3``
        for converting Kpc/:math:`h` to Mpc/:math:`h`.  The velocity offset
        should include the redshift conversion factor (the conformal
        Hubble parameter).
    **kwargs
        Parameters (other than `path` and `names`) to be passed to
        :class:`nbodykit.source.catalog.CSVCatalog`.

    Notes
    -----
    The use of ``offset=True`` assumes the source catalogue contains
    velocity data columns labelled by, for instance, 'vx', for each of
    the position data columns, for instance, 'x'.

    """

    def __init__(self, source_file, headings, mean_density=None, boxsize=None,
                 upscale=1., offset=False, offset_upscale=1.e-3, **kwargs):

        super().__init__(str(source_file), headings, **kwargs)

        # WARNING: Unresolved MPI bug with `nbodykit`.
        if self.comm is not None and self.comm.size > 1:
            warnings.warn(
                "Beware of using multi-processing: "
                "unresolved MPI issue with 'nbodykit'."
            )

        self.attrs['source'] = source_file
        self.attrs['offset'] = offset

        if boxsize is not None:
            self.attrs['BoxSize'] = [boxsize,] * 3 if np.isscalar(boxsize) \
                else boxsize
        else:
            self.attrs['BoxSize'] = \
                np.max(self['Position']) - np.min(self['Position'])

        if mean_density is not None:
            self.attrs['nbar'] = mean_density
        else:
            self.attrs['nbar'] = self.size / np.prod(self.attrs['BoxSize'])

        self['Position'] = \
            self['x'][:, None] * [upscale, 0, 0] \
            + self['y'][:, None] * [0, upscale, 0] \
            + self['z'][:, None] * [0, 0, upscale]

        # Add redshift-space distortions.
        if offset:
            self['Position'] = \
                self['vx'][:, None] * [offset_upscale, 0, 0] \
                + self['vy'][:, None] * [0, offset_upscale, 0] \
                + self['vz'][:, None] * [0, 0, offset_upscale]

        if self.comm is None or self.comm.rank == 0:
            self.logger.debug("%s generated.", self)

    def __str__(self):

        str_info = "nbar={}, boxsize={}, source={}".format(
            self.attrs['nbar'], self.attrs['BoxSize'], self.attrs['source']
        )

        return f"{self.__class__.__name__}({str_info})"


class SphericalFKPCatalogue:
    """FKP-style paired catalogues in a spherical domain.

    Parameters
    ----------
    data_catalogue : :class:`~.SourceCatalogue` *or None, optional*
        Data catalogue.  Cannot be `None` (default) unless `source_file`
        and `source_kwargs` are provided as an alternative to read the
        catalogue source file.
    source_file : str or None, optional
        Catalogue source file path.  This can only be `None` (default) if
        `data_catalogue` is already provided.
    source_kwargs : dict or None, optional
        Parameters to pass to :class:`~.SourceCatalogue`.  This can only
        be `None` (default) if `data_catalogue` is already provided.
    random_catalogue : :class:`nbodykit.base.catalog.CatalogSource` *or None, optional*
        Random catalogue (default is `None`).
    contrast : float or None, optional
        Mean density contrast compared to the data catalogue used to
        generate a random catalogue (default is `None`).  Ignored if
        `random_catalogue` is provided.
    mask : callable or None, optional
        Any veto mask function to be applied to both the data and random
        catalogues.  Must be a function of three Cartesian
        coordinates only assuming the origin is at the centre of
        the catalogues.
    selection : callable or None, optional
        Any selection function (normalised to unity) to be applied to both
        the data and random catalogues.  Must be a function of three
        Cartesian coordinates only assuming the origin is at the centre of
        the catalogues.
    weight : callable or None, optional
        Any weight function to be applied to both the data and random
        catalogues.  Must be a function of three Cartesian
        coordinates only assuming the origin is at the centre of
        the catalogues.
    random_seed : int or None, optional
        Random seed of the random catalogue (default is `None`).
    apply_selection_as_veto : bool, optional
        If `True` (default is `False`), `selection` returning boolean
        values is applied to the 'Selection' column of the catalogues
        instead of the 'NZ' column (see `nbodykit` documentation for
        this peculiarity.)

    Attributes
    ----------
    catalogue_pair : |FKPCatalog|
        FKP-style paired catalogue.
    data_catalogue : :class:`~.SourceCatalogue`
        Data catalogue.
    random_catalogue : :class:`nbodykit.base.catalog.CatalogSource`
        Random catalogue.
    attrs : dict
        Attributes including initialisation parameters.

    Notes
    -----
    If `selection` is provided, it must be normalised to the overall mean
    number density of the data catalogue.


    .. |FKPCatalog| replace::

        :class:`nbodykit.algorithms.convpower.catalog.FKPCatalog`

    """

    _msg = {
        'boxsizes': \
            "Different boxsizes for %s and %s catalogues: %s and %s.",
        'inscribe': (
            "Bounding sphere larger than the %s catalogue: "
            "diameter %.1f > boxsize %s."
        ),
        'centre': (
            "Centred %s catalogue with centred positions in 'Location' column."
        ),
        'sphericalise': "Restricted %s catalogue to spherical domain.",
        'mask': "Applied mask veto to %s catalogue.",
        'selection': "Applied selection function to %s catalogue.",
        'weight': "Applied weighting scheme to %s catalogue.",
    }

    def __init__(self, radius, data_catalogue=None, source_file=None,
                 source_kwargs=None, random_catalogue=None, contrast=None,
                 mask=None, selection=None, weight=None, random_seed=None,
                 apply_selection_as_veto=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        if data_catalogue is None:
            self.data_catalogue = SourceCatalogue(source_file, **source_kwargs)
        else:
            self.data_catalogue = data_catalogue
            try:
                source_file = self.data_catalogue.attrs['source']
            except (AttributeError, KeyError):
                source_file = None

        if random_catalogue is not None:
            self.random_catalogue = random_catalogue
            contrast = \
                random_catalogue.attrs['nbar'] / data_catalogue.attrs['nbar']
        elif contrast is not None:
            self.random_catalogue = RandomCatalogue(
                contrast * self.data_catalogue.attrs['nbar'],
                self.data_catalogue.attrs['BoxSize'],
                seed=random_seed
            )
        else:
            self.random_catalogue = None

        self.attrs = {
            'bounding_radius': radius,
            'source': source_file,
            'contrast': contrast,
        }

        self.catalogue_pair = self._initialise(
            mask, selection, weight, apply_selection_as_veto
        )

    def __str__(self):

        str_info = (
            "radius={bounding_radius}, source={source}, contrast={contrast}"
        ).format(**self.attrs)

        return f"{self.__class__.__name__}({str_info})"

    def _initialise(self, mask, selection, weight, apply_selection_as_veto):

        radius = self.attrs['bounding_radius']

        data_boxsize = self.data_catalogue.attrs['BoxSize']

        if self.random_catalogue is None:
            rand_boxsize = None
        else:
            rand_boxsize = self.random_catalogue.attrs['BoxSize']
            # Compare data and random catalogue boxsizes.
            if not np.allclose(data_boxsize, rand_boxsize):
                warnings.warn(
                    self._msg['boxsizes'],
                    'data', 'random',
                    np.around(data_boxsize, decimals=0),
                    np.around(rand_boxsize, decimals=0)
                )

        # Restrict catalogued in the spherical domain.
        for name, catalogue, boxsize in zip(
                ['data', 'random'],
                [self.data_catalogue, self.random_catalogue],
                [data_boxsize, rand_boxsize]
            ):

            if catalogue is None:
                break

            # Check the bounding sphere is entirely within the catalogue.
            if np.any(np.less(boxsize, 2*radius)):
                self.logger.debug(
                    self._msg['inscribe'], name, 2*radius, boxsize
                )

            # Centre the coordinate origin if not already centred.
            is_origin_at_corner = np.any(
                np.isclose(
                    np.min(np.abs(catalogue['Position']), axis=0)
                    / np.max(np.abs(catalogue['Position']), axis=0),
                    0., atol=1.e-2
                )
            )
            if is_origin_at_corner:
                catalogue['Location'] = \
                    catalogue['Position'] - np.divide(boxsize, 2)
                self.logger.debug(self._msg['centre'], name)
            else:
                catalogue['Location'] = catalogue['Position']

            # Sphericalise and set base selection values ('NZ') for the
            # `nbodykit` `ConvolvedFFTPower` algorithm.
            catalogue['Selection'] &= \
                spherical_indicator(catalogue['Location'], radius)
            self.logger.debug(self._msg['sphericalise'], name)

            # Apply any mask, selection or weight.
            catalogue['NZ'] = \
                self.data_catalogue.attrs['nbar'] * np.ones(catalogue.csize)

            if callable(mask):
                catalogue['Selection'] &= mask(catalogue['Location'])
                self.logger.debug(self._msg['mask'], name)

            if callable(selection):
                if apply_selection_as_veto:
                    catalogue['Selection'] *= selection(catalogue['Location'])
                else:
                    catalogue['NZ'] *= selection(catalogue['Location'])
                self.logger.debug(self._msg['selection'], name)

            if callable(weight):
                catalogue['Weight'] *= weight(catalogue['Location'])
                self.logger.debug(self._msg['weight'], name)

        # Construct FKP catalogue.
        catalogue_pair = FKPCatalog(self.data_catalogue, self.random_catalogue)

        return catalogue_pair
