"""
Catalogue maker (:mod:`~harmonia.mapper.catalogue_maker`)
===============================================================================

Make discrete catalogues from observed or simulated realisations with
:mod:`nbodykit.lab` implementation.

.. autosummary::

    spherical_indicator
    RandomCatalogue
    LognormalCatalogue

|

.. todo:: Under further development if ``nbodykit`` is to be subtituted.

"""
import logging

import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.base.catalog import CatalogSource, column
from nbodykit.lab import LogNormalCatalog, UniformCatalog

from harmonia.algorithms import (
    generate_gaussian_random_field as gen_gaussian_field,
    generate_lognormal_random_field as gen_lognormal_field,
    poisson_sample as smp_field,
    populate_particles as pop_field,
    )
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
    return np.linalg.norm(cartesian_position, axis=-1) <= bounding_radius


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
            Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
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


# TODO: Implement RSD.
class GaussianCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum.

    Attributes
    ----------
    comm : :class:`nbodykit.CurrentMPIComm`
        Current MPI communicator.
    Plin : callable
        Input linear matter power spectrum (in cubic Mpc/h).
    attrs : dict
        Generic attributes.

    Warnings
    --------

    The field statistics may change after Poisson sampling to discrete
    particles as the density contrast is clipped below at -1.

    """

    _logger = logging.getLogger("GaussianCatalogue")

    @CurrentMPIComm.enable
    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., seed=None,
                 comm=None):
        """
        Parameters
        ----------
        Plin : callable
            Linear matter power spectrum (in cubic Mpc/h).
        nbar : float
            Desired mean particle number density (in cubic h/Mpc).
        BoxSize : float
            Catalogue box size per dimension (in Mpc/h).
        Nmesh : int
            Mesh grid number per dimension.
        bias : float, optional
            Particle bias relative to the input power spectrum (default is 2.).
        seed : int or None, optional
            Random seed of the catalogue (default is `None`).
        comm : :class:`nbodykit.CurrentMPIComm` or None, optional
            Current MPI communicator.

        """
        self.comm = comm
        self.Plin = Plin

        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)

        if hasattr(Plin, 'attrs'):
            self.attrs.update(Plin.attrs)

        self.attrs.update({
            'nbar': nbar,
            'bias': bias,
            'BoxSize': [BoxSize]*3,
            'Nmesh': Nmesh,
            'seed': seed,
            })

        # Generate fields.
        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        gaussian_field = gen_gaussian_field(
            BoxSize, Nmesh, Plin, bias=bias, seed=field_seed
            )
        sampled_field = smp_field(
            gaussian_field, nbar, BoxSize, seed=sampling_seed
            )
        position = pop_field(sampled_field, nbar, BoxSize, seed=drift_seed)

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)

    def __repr__(self):
        return (
            "GaussianCatalogue"
            "(nmean={0}, bias={1}, RSD={2}, boxsize={3}, nmesh={4}, seed={5})"
            ).format(
                self.attrs['nbar'], self.attrs['bias'], '',
                self.attrs['BoxSize'], self.attrs['Nmesh'], self.attrs['seed']
                )

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)


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
            Input mean particle number density (lenght unit Mpc/h).
        boxsize : float, array_like
            Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
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


# TODO: Implement RSD.
class LogNormalCatalogue(CatalogSource):
    """Log-normal random catalogue of given number density and power spectrum.

    Attributes
    ----------
    comm : :class:`nbodykit.CurrentMPIComm`
        Current MPI communicator.
    Plin : callable
        Input linear matter power spectrum (in cubic Mpc/h).
    attrs : dict
        Generic attributes.

    """

    _logger = logging.getLogger("LogNormalCatalogue")

    @CurrentMPIComm.enable
    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., seed=None,
                 comm=None):
        """
        Parameters
        ----------
        Plin : callable
            Linear matter power spectrum (in cubic Mpc/h).
        nbar : float
            Desired mean particle number density (in cubic h/Mpc).
        BoxSize : float
            Catalogue box size per dimension (in Mpc/h).
        Nmesh : int
            Mesh grid number per dimension.
        bias : float, optional
            Particle bias relative to the input power spectrum (default is 2.).
        seed : int or None, optional
            Random seed of the catalogue (default is `None`).
        comm : :class:`nbodykit.CurrentMPIComm` or None, optional
            Current MPI communicator.

        """
        self.Plin = Plin
        self.comm = comm

        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)

        if hasattr(Plin, 'attrs'):
            self.attrs.update(Plin.attrs)

        self.attrs.update({
            'nbar': nbar,
            'bias': bias,
            'BoxSize': [BoxSize]*3,
            'Nmesh': Nmesh,
            'seed': seed
            })

        # Generate fields.
        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        lognormal_field = gen_lognormal_field(
            BoxSize, Nmesh, Plin, bias=bias, seed=field_seed
            )
        sampled_field = smp_field(
            lognormal_field, nbar, BoxSize, seed=sampling_seed
            )
        position = pop_field(sampled_field, nbar, BoxSize, seed=drift_seed)

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)

    def __repr__(self):
        return (
            "LogNormalCatalogue"
            "(nmean={0}, bias={1}, RSD={2}, boxsize={3}, nmesh={4}, seed={5})"
            ).format(
                self.attrs['nbar'], self.attrs['bias'], '',
                self.attrs['BoxSize'], self.attrs['Nmesh'], self.attrs['seed']
                )

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)
