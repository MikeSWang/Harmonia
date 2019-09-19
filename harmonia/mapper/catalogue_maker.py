"""
Catalogue maker (:mod:`~harmonia.mapper.catalogue_maker`)
===============================================================================

Make discrete catalogues from observed or simulated realisations.

.. autosummary::

    spherical_indicator
    RandomCatalogue
    LognormalCatalogue
    LogNormalCatalogue
    GaussianCatalogue

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
            self._logger.info("RSDs added to radial particle positions. ")

    def __repr__(self):
        return (
            "LognormalCatalogue"
            "(nmean={0}, bias={1}, RSD={2}, boxsize={3}, nmesh={4}, seed={5})"
            ).format(
                self.attrs['nbar'], self.attrs['bias'], self.attrs['RSD'],
                self.attrs['BoxSize'], self.attrs['Nmesh'], self.attrs['seed']
                )


class LogNormalCatalogue(CatalogSource):
    """Log-normal random catalogue of given number density and power spectrum
    with particle velocities predicted by the Zel'dovich approximation.

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
    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., add_RSD=False,
                 growth_rate=None, los=None, seed=None, comm=None):
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
        add_RSD : bool, optional
            If `True` (default is `False`), add appropriately normalised
            redshift-space velocity offset to particle positions.
        growth_rate : float or None, optional
            If `add_RSD` is `True` and `Plin` does not have both 'cosmo' and
            'redshift' attributes, then this cannot be `None` (default).
        los : tuple or list [of length 3] or (3,) array_like or None, optional
            Line-of-sight direction vector.  If `None` (default), this is set
            to the radial directions.
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
            'BoxSize': [BoxSize,]*3,
            'Nmesh': Nmesh,
            'RSD': add_RSD,
            'seed': seed
            })

        if hasattr(Plin, 'cosmo') and hasattr(Plin, 'redshift'):
            cosmo = getattr(Plin, 'cosmo', None)
            redshift = getattr(Plin, 'redshift', None)
            growth_rate = cosmo.scale_independent_growth_rate(redshift)
        elif add_RSD and growth_rate is None:
            raise ValueError(
                "`growth_rate` cannot be None if `add_RSD` is True and "
                "'cosmo' and 'redshift' attributes are absent in `Plin`. "
                )
        elif not add_RSD:
            growth_rate = None
        self.attrs['growth_rate'] = growth_rate

        # Generate fields.
        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        if not add_RSD:
            lognormal_field = gen_lognormal_field(
                BoxSize, Nmesh, Plin, bias=bias, seed=field_seed
                )
        else:
            lognormal_field, lognormal_vfield = gen_lognormal_field(
                BoxSize, Nmesh, Plin, bias=bias, retdisp=True, seed=field_seed
                )
        sampled_field = smp_field(
            lognormal_field, nbar, BoxSize, seed=sampling_seed
            )
        if not add_RSD:
            position = pop_field(sampled_field, nbar, BoxSize, seed=drift_seed)
        else:
            position, displacement = pop_field(
                sampled_field, nbar, BoxSize, voff_fields=lognormal_vfield,
                seed=drift_seed
                )

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)
        self._logger.info("%s generated. ", self.__repr__())

        # Add redshift-space distortions,
        if add_RSD:
            self._veloff = growth_rate * displacement
            if los is None:  # radial distortion only
                self['Position'] += self['VelocityOffset'] \
                    * normalise_vector(self['Position'])
            else:  # forced line-of-sight direction
                self['Position'] += self['VelocityOffset'] * np.array(los)
            self._logger.info("RSDs added to radial particle positions. ")

    def __repr__(self):
        return (
            "LogNormalCatalogue"
            "(nmean={0}, bias={1}, fRSD={2}, boxsize={3}, nmesh={4}, seed={5})"
            ).format(
                self.attrs['nbar'], self.attrs['bias'],
                self.attrs['growth_rate'], self.attrs['BoxSize'],
                self.attrs['Nmesh'], self.attrs['seed']
                )

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)

    @column
    def VelocityOffset(self):
        """Particle velocity offsets to positions appropriately normalised (in
        Mpc/h).

        """
        return self.make_column(self._veloff)


class GaussianCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum
    with particle velocities predicted by the Zel'dovich approximation.

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
    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., add_RSD=False,
                 growth_rate=None, los=None, seed=None, comm=None):
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
        add_RSD : bool, optional
            If `True` (default is `False`), add appropriately normalised
            redshift-space velocity offset to particle positions.
        growth_rate : float or None, optional
            If `add_RSD` is `True` and `Plin` does not have both 'cosmo' and
            'redshift' attributes, then this cannot be `None` (default).
        los : tuple or list [of length 3] or (3,) array_like or None, optional
            Line-of-sight direction vector.  If `None` (default), this is set
            to the radial directions.
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
            'BoxSize': [BoxSize,]*3,
            'Nmesh': Nmesh,
            'RSD': add_RSD,
            'seed': seed,
            })

        if add_RSD and hasattr(Plin, 'cosmo') and hasattr(Plin, 'redshift'):
            cosmo = getattr(Plin, 'cosmo', None)
            redshift = getattr(Plin, 'redshift', None)
            growth_rate = cosmo.scale_independent_growth_rate(redshift)
        elif add_RSD and growth_rate is None:
            raise ValueError(
                "`growth_rate` cannot be None if `add_RSD` is True and "
                "'cosmo' and 'redshift' attributes are absent in `Plin`. "
                )
        elif not add_RSD:
            growth_rate = None
        self.attrs['growth_rate'] = growth_rate

        # Generate fields.
        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        if not add_RSD:
            gaussian_field = gen_gaussian_field(
                BoxSize, Nmesh, Plin, bias=bias, seed=field_seed
                )
        else:
            gaussian_field, gaussian_vfield = gen_gaussian_field(
                BoxSize, Nmesh, Plin, bias=bias, retdisp=True, seed=field_seed
                )
        sampled_field = smp_field(
            gaussian_field, nbar, BoxSize, seed=sampling_seed
            )
        if not add_RSD:
            position = pop_field(sampled_field, nbar, BoxSize, seed=drift_seed)
        else:
            position, displacement = pop_field(
                sampled_field, nbar, BoxSize, voff_fields=gaussian_vfield,
                seed=drift_seed
                )

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)
        self._logger.info("%s generated. ", self.__repr__())

        # Add redshift-space distortions,
        if add_RSD:
            self._veloff = growth_rate * displacement
            if los is None:  # radial distortion only
                self['Position'] += self['VelocityOffset'] \
                    * normalise_vector(self['Position'])
            else:  # forced line-of-sight direction
                self['Position'] += self['VelocityOffset'] * np.array(los)
            self._logger.info("RSDs added to radial particle positions. ")

    def __repr__(self):
        return (
            "GaussianCatalogue"
            "(nmean={0}, bias={1}, fRSD={2}, boxsize={3}, nmesh={4}, seed={5})"
            ).format(
                self.attrs['nbar'], self.attrs['bias'],
                self.attrs['growth_rate'], self.attrs['BoxSize'],
                self.attrs['Nmesh'], self.attrs['seed']
                )

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)

    @column
    def VelocityOffset(self):
        """Particle velocity offsets to positions appropriately normalised (in
        Mpc/h).

        """
        return self.make_column(self._veloff)
