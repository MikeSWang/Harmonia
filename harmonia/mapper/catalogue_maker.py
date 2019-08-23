"""
Catalogue maker (:mod:`~harmonia.mapper.catalogue_maker`)
===============================================================================

Make discrete catalogues from observed or simulated realisations with
:mod:`nbodykit.lab` implementation.

.. todo:: Under further development if ``nbodykit`` is to be subtituted.

"""
import logging

from numpy.linalg import norm
from nbodykit.lab import LogNormalCatalog, UniformCatalog

from harmonia.collections import normalise_vector


def spherical_indicator(cartesian_position, bounding_radius):
    """Indicate whether an object lies within a spherical domain.

    Parameters
    ----------
    cartesian_position : float, array_like
        Object position in Cartesian coordinates.
    bounding_radius : float
        Radius of the bounding sphere.

    Returns
    -------
    bool, array_like
        `True` if the object lies within the spherical domain.

    """
    return norm(cartesian_position, axis=-1) <= bounding_radius


class RandomCatalogue(UniformCatalog):
    """Uniform random catalogue of given mean particle number density and box
    size.

    """

    _logger = logging.getLogger('RandomCatalogue')

    def __init__(self, nmean, boxsize, seed=None):
        """
        Parameters
        ----------
        nmean : float
            Input mean particle number density.
        boxsize : float, array_like
            Catalogue box size as a scalar or a triple of scalars.
        seed : int or None, optional
            Random seed of the catalogue (default is `None`).

        """
        UniformCatalog.__init__(self, nmean, boxsize, seed=seed)
        self.attrs['nbar'] = nmean
        self._logger.debug("%s generated. ", self.__repr__())

    def __repr__(self):
        return "RandomCatalogue(nmean={0}, boxsize={1}, seed={2})".format(
            self.attrs['nbar'], self.attrs['BoxSize'], self.attrs['seed']
            )


class LognormalCatalogue(LogNormalCatalog):
    """Log-normal random catalogue of given linear power spectrum with particle
    velocities predicted by the Zel'dovich approximation.

    """

    _logger = logging.getLogger('LognormalCatalogue')

    def __init__(self, powerlin, nmean, boxsize, nmesh, bias=2., add_RSD=False,
                 seed=None):
        """
        Parameters
        ----------
        powerlin : callable
            Linear matter power spectrum with specified cosmology and redshift.
        nmean : float
            Input mean particle number density.
        boxsize : float, array_like
            Catalogue box size as a scalar or a triple of scalars.
        nmesh : int
            Mesh grid number for FFT generation.
        bias : float, optional
            Particle bias relative to the matter distribution (default is 2.).
        add_RSD : bool, optional
            If `True` (default is `False`), add appropriately normalised
            redshift-space velocity offset to particle positions.
        seed : int or None, optional
            Random seed of the catalogue (default is `None`).

        """
        LogNormalCatalog.__init__(
            self, powerlin, nmean, boxsize, nmesh, bias=bias, seed=seed
            )
        self.attrs['RSD'] = add_RSD
        self._logger.info("%s generated. ", self.__repr__())

        if add_RSD:
            self['Position'] += self['VelocityOffset'] \
                * normalise_vector(self['Position'])  # radial distortion only
            self._logger.info("RSDs added to radial particle velocities. ")

    def __repr__(self):
        return (
            "LognormalCatalogue"
            "(nmean={0}, bias={1}, RSD={2}, boxsize={3}, nmesh={4}, seed={5})"
            ).format(
                self.attrs['nbar'], self.attrs['bias'], self.attrs['RSD'],
                self.attrs['BoxSize'], self.attrs['Nmesh'], self.attrs['seed']
                )
