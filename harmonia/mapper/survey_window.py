"""
Survey window (:mod:`~harmonia.mapper.survey_window`)
===========================================================================

Determine the window function from survey specifications with high-density
random catalogues.

.. autosummary::

    WindowFunction

|

"""
import numpy as np
from nbodykit.lab import SurveyDataPairCount as PairCount
from nbodykit.transform import CartesianToSky

from harmonia.cosmology import fiducial_cosmology
from harmonia.collections.utils import cartesian_to_spherical as c2s
from .catalogue_maker import RandomCatalogue


class WindowFunction:
    """Window function in configuration space.

    Parameters
    ----------
    mean_density : float
        Desired mean particle number density (in cubic h/Mpc).
    boxsize : float, array_like
        Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
    mask, selection, weight : callable or None, optional
        Specification functions (default is `None`).
    source : {'mock', 'survey'}, optional
        Source type of the catalogue.  If ``'mock'`` (default), `mask`,
        `selection` and `weight` are assumed to be functions of Cartesian
        coordinates whose origin is the centre of the catalogue; if
        ``'survey'``, `mask`, `selection` and `weight` are assumed to be
        functions of spherical coordinates.

    """

    def __init__(self, mean_density, boxsize, mask=None, selection=None,
                 weight=None, source='mock'):

        self.mask, self.selection, self.weight = mask, selection, weight

        self._nbar = mean_density
        self._boxsize = boxsize
        self._source = source

        self._catalogue_ = None

    @property
    def _catalogue(self):

        if self._catalogue_ is not None:
            return self._catalogue_

        self._catalogue_ = RandomCatalogue(self._nbar, self._boxsize)

        if self._source == 'mock':
            self._catalogue_['Location'] = \
                self._catalogue_['Position'] - np.divide(self._boxsize, 2)
        elif self._source == 'survey':
            self._catalogue_['Location'] = c2s(
                self._catalogue_['Position'] - np.divide(self._boxsize, 2)
            )

        if callable(self.mask):
            self._catalogue_['Weight'] *= \
                self.mask(self._catalogue_['Location'])
        if callable(self.selection):
            self._catalogue_['Weight'] *= \
                self.selection(self._catalogue_['Location'])
        if callable(self.weight):
            self._catalogue_['Weight'] *= \
                self.weight(self._catalogue_['Location'])

        return self._catalogue_

    def _pair_counts(self, edges):

        ra, dec, z = CartesianToSky(
            self._catalogue['Location'], fiducial_cosmology
        )

        self._catalogue['RA'] = ra
        self._catalogue['DEC'] = dec
        self._catalogue['Redshift'] =z

        self._pair_counts = PairCount(
            '2d', self.catalogue, edges, cosmo=fiducial_cosmology, Nmu=None
        ).pairs

        return self._pair_counts

    def window_multipoles(self, poles, edges):
        """Determine window function multipoles from pair counts of a
        random catalogue with the specified window.

        """