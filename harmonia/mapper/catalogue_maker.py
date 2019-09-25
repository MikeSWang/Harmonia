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

    Parameters
    ----------
    mean_density : float
        Input mean particle number density (in cubic h/Mpc).
    boxsize : float, array_like
        Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).

    """

    _logger = logging.getLogger('RandomCatalogue')

    def __init__(self, mean_density, boxsize, seed=None):
        UniformCatalog.__init__(self, mean_density, boxsize, seed=seed)
        self.attrs['nbar'] = mean_density
        self._logger.debug("%s generated. ", self.__str__())

    def __str__(self):
        return "RandomCatalogue(nbar={0}, boxsize={1}, seed={2})".format(
            self.attrs['nbar'],
            self.attrs['BoxSize'],
            self.attrs['seed'],
        )


class LognormalCatalogue(LogNormalCatalog):
    """Log-normal random catalogue of given linear power spectrum with particle
    velocities predicted by the Zel'dovich approximation.

    This is the direct implementation by
    :class:`nbodykit.source.catalog.lognormal.LogNormalCatalog`.

    Parameters
    ----------
    Plin : callable
        Linear matter power spectrum with specified cosmology and redshift.
    nbar : float
        Input mean particle number density (lenght unit Mpc/h).
    BoxSize : float, array_like
        Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
    Nmesh : int
        Mesh grid number for FFT generation.
    bias : float, optional
        Particle bias relative to the matter distribution (default is 2.).
    add_RSD : bool, optional
        If `True` (default is `False`), add appropriately normalised
        redshift-space velocity offset to particle positions.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).

    """

    _logger = logging.getLogger('LognormalCatalogue')

    def __init__(self, Plin, nbar, BoxSize, Nmesh, bias=2., add_RSD=False,
                 seed=None):
        super().__init__(
            self,
            Plin,
            nbar,
            BoxSize,
            Nmesh,
            bias=bias,
            seed=seed,
        )
        self.attrs['RSD_flag'] = add_RSD
        self._logger.info("%s generated. ", self.__str__())
        if add_RSD:
            self['Position'] += self['VelocityOffset'] \
                * normalise_vector(self['Position'])
            self._logger.info("RSDs added to radial particle positions. ")

    def __str__(self):
        return (
            "LognormalCatalogue"
            "(nbar={0}, bias={1}, RSD={2}, BoxSize={3}, Nmesh={4}, seed={5})"
        ).format(
            self.attrs['nbar'],
            self.attrs['bias'],
            self.attrs['RSD_flag'],
            self.attrs['BoxSize'],
            self.attrs['Nmesh'],
            self.attrs['seed'],
        )


class LogNormalCatalogue(CatalogSource):
    """Log-normal random catalogue of given number density and power spectrum
    with particle velocities predicted by the Zel'dovich approximation.

    Parameters
    ----------
    power_spectrum : callable
        Linear matter power spectrum (in cubic Mpc/h).
    mean_density : float
        Desired mean particle number density (in cubic h/Mpc).
    boxsize : float
        Catalogue box size per dimension (in Mpc/h).
    num_mesh : int
        Mesh grid number per dimension.
    bias : float, optional
        Particle bias relative to the input power spectrum (default is 2.).
    add_RSD : bool, optional
        If `True` (default is `False`), add appropriately normalised
        redshift-space velocity offset to particle positions.
    growth_rate : float or None, optional
        If `add_RSD` is `True` and `Plin` does not have both 'cosmo' and
        'redshift' attributes, then this cannot be `None` (default).
    line_of_sight : tuple or list [of length 3], array_like or None, optional
        Line-of-sight direction vector.  If `None` (default), this is set
        to the radial directions.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).
    comm : :class:`nbodykit.CurrentMPIComm` or None, optional
        Current MPI communicator.

    """

    _logger = logging.getLogger("LogNormalCatalogue")

    @CurrentMPIComm.enable
    def __init__(self, power_spectrum, mean_density, boxsize, num_mesh,
                 bias=2., add_RSD=False, growth_rate=None, line_of_sight=None,
                 seed=None, comm=None):

        # Get attributes.
        self.power_spectrum = power_spectrum
        self.comm = comm
        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)

        if hasattr(power_spectrum, 'attrs'):
            self.attrs.update(power_spectrum.attrs)
        self.attrs.update(
            {
                'nbar': mean_density,
                'bias': bias,
                'BoxSize': [boxsize,]*3,
                'Nmesh': num_mesh,
                'RSD_flag': add_RSD,
                'seed': seed,
            }
        )

        if hasattr(power_spectrum, 'cosmo') \
                and hasattr(power_spectrum, 'redshift'):
            cosmo = getattr(power_spectrum, 'cosmo', None)
            redshift = getattr(power_spectrum, 'redshift', None)
            growth_rate = cosmo.scale_independent_growth_rate(redshift)
        elif add_RSD and (growth_rate is None):
            raise ValueError(
                "`growth_rate` cannot be None if `add_RSD` is True and "
                "'cosmo' and 'redshift' attributes are absent "
                "in `power_spectrum`. "
            )
        elif not add_RSD:
            growth_rate = None
        self.attrs['growth_rate'] = growth_rate

        # Generate fields.
        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        field, vec_field = gen_lognormal_field(
            boxsize,
            num_mesh,
            power_spectrum,
            bias=bias,
            return_disp=add_RSD,
            seed=field_seed,
        )
        sampled_field = smp_field(
            field,
            mean_density,
            boxsize,
            seed=sampling_seed,
            )
        position, displacement = pop_field(
            sampled_field,
            mean_density,
            boxsize,
            vel_offset_fields=vec_field,
            seed=drift_seed,
        )

        # Instantiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)
        self._logger.info("%s generated. ", self.__str__())

        if add_RSD:
            self._vel_offset = growth_rate * displacement
            if line_of_sight is None:  # radial distortion only
                self['Position'] += self['VelocityOffset'] \
                    * normalise_vector(self['Position'])
            else:  # forced line-of-sight direction
                self['Position'] += self['VelocityOffset'] \
                    * np.array(line_of_sight)
            self._logger.info("RSDs added to radial particle positions. ")

    def __str__(self):
        return (
            "LogNormalCatalogue"
            "(nbar={0}, bias={1}, f={2}, boxsize={3}, num_mesh={4}, seed={5})"
        ).format(
            self.attrs['nbar'],
            self.attrs['bias'],
            self.attrs['growth_rate'],
            self.attrs['BoxSize'],
            self.attrs['Nmesh'],
            self.attrs['seed'],
        )

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)

    @column
    def VelocityOffset(self):
        """Appropriately normalised particle velocity offsets (in Mpc/h) to
        positions.

        """
        return self.make_column(self._vel_offset)


class GaussianCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum
    with particle velocities predicted by the Zel'dovich approximation.

    Parameters
    ----------
    power_spectrum : callable
        Linear matter power spectrum (in cubic Mpc/h).
    mean_density : float
        Desired mean particle number density (in cubic h/Mpc).
    boxsize : float
        Catalogue box size per dimension (in Mpc/h).
    num_mesh : int
        Mesh grid number per dimension.
    bias : float, optional
        Particle bias relative to the input power spectrum (default is 2.).
    add_RSD : bool, optional
        If `True` (default is `False`), add appropriately normalised
        redshift-space velocity offset to particle positions.
    growth_rate : float or None, optional
        If `add_RSD` is `True` and `Plin` does not have both 'cosmo' and
        'redshift' attributes, then this cannot be `None` (default).
    line_of_sight : tuple or list [of length 3], array_like or None, optional
        Line-of-sight direction vector.  If `None` (default), this is set
        to the radial directions.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).
    comm : :class:`nbodykit.CurrentMPIComm` or None, optional
        Current MPI communicator.

    Warnings
    --------
    The field statistics may change after Poisson sampling to discrete
    particles as the density contrast is clipped below at -1.

    """

    _logger = logging.getLogger("GaussianCatalogue")

    @CurrentMPIComm.enable
    def __init__(self, power_spectrum, mean_density, boxsize, num_mesh,
                 bias=2., add_RSD=False, growth_rate=None, line_of_sight=None,
                 seed=None, comm=None):

        # Get attributes.
        self.power_spectrum = power_spectrum
        self.comm = comm
        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)

        if hasattr(power_spectrum, 'attrs'):
            self.attrs.update(power_spectrum.attrs)
        self.attrs.update(
            {
                'nbar': mean_density,
                'bias': bias,
                'BoxSize': [boxsize,]*3,
                'Nmesh': num_mesh,
                'RSD_flag': add_RSD,
                'seed': seed,
            }
        )

        if hasattr(power_spectrum, 'cosmo') \
                and hasattr(power_spectrum, 'redshift'):
            cosmo = getattr(power_spectrum, 'cosmo', None)
            redshift = getattr(power_spectrum, 'redshift', None)
            growth_rate = cosmo.scale_independent_growth_rate(redshift)
        elif add_RSD and (growth_rate is None):
            raise ValueError(
                "`growth_rate` cannot be None if `add_RSD` is True and "
                "'cosmo' and 'redshift' attributes are absent "
                "in `power_spectrum`. "
            )
        elif not add_RSD:
            growth_rate = None
        self.attrs['growth_rate'] = growth_rate

        # Generate fields.
        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        field, vec_field = gen_gaussian_field(
            boxsize,
            num_mesh,
            power_spectrum,
            bias=bias,
            return_disp=add_RSD,
            seed=field_seed,
        )
        sampled_field = smp_field(
            field,
            mean_density,
            boxsize,
            seed=sampling_seed,
            )
        position, displacement = pop_field(
            sampled_field,
            mean_density,
            boxsize,
            vel_offset_fields=vec_field,
            seed=drift_seed,
        )

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)
        self._logger.info("%s generated. ", self.__str__())

        # Add redshift-space distortions,
        if add_RSD:
            self._vel_offset = growth_rate * displacement
            if line_of_sight is None:  # radial distortion only
                self['Position'] += self['VelocityOffset'] \
                    * normalise_vector(self['Position'])
            else:  # forced line-of-sight direction
                self['Position'] += self['VelocityOffset'] \
                    * np.array(line_of_sight)
            self._logger.info("RSDs added to radial particle positions. ")

    def __str__(self):
        return (
            "GaussianCatalogue"
            "(nbar={0}, bias={1}, fRSD={2}, boxsize={3}, Nmesh={4}, seed={5})"
        ).format(
            self.attrs['nbar'],
            self.attrs['bias'],
            self.attrs['growth_rate'],
            self.attrs['BoxSize'],
            self.attrs['Nmesh'],
            self.attrs['seed'],
        )

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)

    @column
    def VelocityOffset(self):
        """Appropriately normalised particle velocity offsets (in Mpc/h) to
        positions.

        """
        return self.make_column(self._vel_offset)
