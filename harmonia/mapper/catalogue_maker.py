"""
Catalogue maker (:mod:`~harmonia.mapper.catalogue_maker`)
===========================================================================

Make discrete catalogues from observed or simulated realisations.

.. autosummary::

    load_catalogue_from_file
    RandomCatalogue
    NBKCatalogue
    LogNormalCatalogue
    GaussianCatalogue

.. warning::

    :class:`~harmonia.mapper.catalogue_maker.LogNormalCatalogue` and
    :class:`~harmonia.mapper.catalogue_maker.GaussianCatalogue` are
    experimental catalogue generators in replacement of
    :class:`nbodykit.source.catalog.lognormal.LogNormalCatalog`.  Use
    with caution.

"""
import logging

import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.base.catalog import CatalogSource, column
from nbodykit.lab import CSVCatalog, LogNormalCatalog, UniformCatalog

from harmonia.algorithms.fields import \
    generate_gaussian_random_field as gen_gaussian_field
from harmonia.algorithms.fields import \
    generate_lognormal_random_field as gen_lognormal_field
from harmonia.algorithms.fields import poisson_sample as smp_field
from harmonia.algorithms.fields import populate_particles as pop_field
from harmonia.collections.utils import normalise_vector

_MAX_INT = 4294967295


def load_catalogue_from_file(file_path, headings, boxsize, unit_scale=1.,
                             add_vel=False, vel_offset_unit_scale=1.e-3):
    """Load catalogue from a file.

    Parameters
    ----------
    file_path : str
        Catalogue file path.
    headings : list of str
        Column headings of the catalogue file.
    boxsize : float
        Catalogue box size.
    unit_scale : float, optional
        Scaling factor for converting the length unit to Mpc/:math:`h`
        (default is 1.), e.g. ``unit_scale = 1.e-3`` for converting
        Kpc/:math:`h` to Mpc/:math:`h`.
    add_vel : bool, optional
        If `True` (default is `False`), add the velocity columns to
        position columns (for e.g. redshift-space distortions).
    vel_offset_unit_scale : float, optional
        Scaling factor for converting the velocity offset length unit to
        Mpc/:math:`h` (default is 1.e-3), e.g.
        ``vel_offset_unit_scale = 1.e-3`` for converting Kpc/:math:`h` to
        Mpc/:math:`h`.  The velocity offset should include the redshift
        conversion factor (the conformal Hubble parameter).

    Returns
    -------
    catalogue : :class:`nbodykit.base.catalog.CatalogSource`
        Catalogue object.

    """
    catalogue = CSVCatalog(file_path, headings)

    catalogue.attrs['BoxSize'] = boxsize

    catalogue['Position'] = \
        catalogue['x'][:, None] * [unit_scale, 0, 0] \
        + catalogue['y'][:, None] * [0, unit_scale, 0] \
        + catalogue['z'][:, None] * [0, 0, unit_scale]

    if add_vel:
        catalogue['Position'] = \
            catalogue['vx'][:, None] * [vel_offset_unit_scale, 0, 0] \
            + catalogue['vy'][:, None] * [0, vel_offset_unit_scale, 0] \
            + catalogue['vz'][:, None] * [0, 0, vel_offset_unit_scale]

    return catalogue


class RandomCatalogue(UniformCatalog):
    """Uniform random catalogue of given mean particle number density and
    box size.

    Parameters
    ----------
    mean_density : float
        Desired mean particle number density (in cubic :math:`h`/Mpc).
    boxsize : float, array_like
        Catalogue box size (in Mpc/:math:`h`) as a scalar or a triple of
        scalars.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).

    """

    _logger = logging.getLogger('RandomCatalogue')

    def __init__(self, mean_density, boxsize, seed=None, comm=None):

        super().__init__(mean_density, boxsize, seed=seed, comm=comm)

        self.attrs['nbar'] = mean_density

        if self.comm is None or self.comm.rank == 0:
            self._logger.debug("%s generated. ", self.__str__())

    def __str__(self):

        return "RandomCatalogue(nbar={0}, boxsize={1}, seed={2})".format(
            self.attrs['nbar'], self.attrs['BoxSize'], self.attrs['seed']
        )


class NBKCatalogue(LogNormalCatalog):
    """``nbodykit`` log-normal random catalogue of given linear power
    spectrum with particle velocitie displacements predicted by the
    Zel'dovich approximation.

    Parameters
    ----------
    power_spectrum : :class:`nbodykit.cosmology.power.linear.LinearPower`
        Desired linear matter power spectrum with specified cosmology and
        redshift.
    mean_density : float
        Desired mean particle number density (in cubic :math:`h`/Mpc).
    boxsize : float, array_like
        Catalogue box size (in Mpc/:math:`h`) as a scalar or a triplet of
        scalars.
    num_mesh : int
        Mesh grid number per dimension for FFT generation.
    bias : float, optional
        Desired particle bias relative to the matter distribution (default
        is 2.).
    add_RSD : bool, optional
        If `True` (default is `False`), add appropriately normalised
        redshift-space velocity offset to particle positions.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).

    """

    _logger = logging.getLogger('LognormalCatalogue')

    def __init__(self, power_spectrum, mean_density, boxsize, num_mesh,
                 bias=2., add_RSD=False, seed=None, comm=None):

        ini_args = (power_spectrum, mean_density, boxsize, num_mesh)

        super().__init__(*ini_args, bias=bias, seed=seed, comm=comm)

        self.attrs['RSD_flag'] = add_RSD
        if self.comm is None or self.comm.rank == 0:
            self._logger.info("%s generated. ", self.__str__())

        if self.attrs['RSD_flag']:
            self['Position'] += self['VelocityOffset'] \
                * normalise_vector(self['Position'])
            if self.comm is None or self.comm.rank == 0:
                self._logger.info("RSDs added to radial particle positions. ")

    def __str__(self):

        str_root = (
            "LognormalCatalogue"
            "(nbar={0}, bias={1}, RSD={2}, BoxSize={3}, Nmesh={4}, seed={5})"
        )
        str_args = (
            self.attrs['nbar'],
            self.attrs['bias'],
            self.attrs['RSD_flag'],
            self.attrs['BoxSize'],
            self.attrs['Nmesh'],
            self.attrs['seed'],
        )

        return str_root.format(*str_args)


class LogNormalCatalogue(CatalogSource):
    """Log-normal random catalogue of given number density and power
    spectrum with particle velocities fulfilling the continuity equation.

    Parameters
    ----------
    power_spectrum : callable
        Desired linear matter power spectrum (in cubic Mpc/:math:`h`).
    mean_density : float
        Desired mean particle number density (in cubic :math:`h`/Mpc).
    boxsize : float
        Catalogue box size per dimension (in Mpc/:math:`h`).
    num_mesh : int
        Mesh grid number per dimension for FFT generation.
    bias : float, optional
        Desired particle bias relative to the matter distribution (default
        is 2.).
    add_RSD : bool, optional
        If `True` (default is `False`), add appropriately normalised
        redshift-space velocity offset to particle positions.
    growth_rate : float or None, optional
        Linear growth rate.  Cannot be `None` (default) if `add_RSD` is
        `True` and `Plin` does not have both 'cosmo' and 'redshift'
        attributes.
    line_of_sight : array_like or None, optional
        Line-of-sight direction vector.  If `None` (default), this is set
        to the radial directions.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).
    comm : :class:`nbodykit.CurrentMPIComm` *or None, optional*
        Current MPI communicator.

    """

    _logger = logging.getLogger("LogNormalCatalogue")

    @CurrentMPIComm.enable
    def __init__(self, power_spectrum, mean_density, boxsize, num_mesh,
                 bias=2., add_RSD=False, growth_rate=None, line_of_sight=None,
                 seed=None, comm=None):

        self.comm = comm
        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, _MAX_INT)
            seed = self.comm.bcast(seed)

        self.power_spectrum = power_spectrum
        if hasattr(power_spectrum, 'attrs'):
            self.attrs.update(power_spectrum.attrs)

        self.attrs.update(
            {
                'nbar': mean_density,
                'bias': bias,
                'BoxSize': [boxsize] * 3,
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
        elif growth_rate is None and self.attrs['RSD_flag']:
            raise ValueError(
                "`growth_rate` cannot be None if `add_RSD` is True and "
                "`power_spectrum` does not have 'cosmo' and 'redshift' "
                "attributes. "
            )
        elif not self.attrs['RSD_flag']:
            growth_rate = None
        self.attrs['growth_rate'] = growth_rate

        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        field, vec_field = gen_lognormal_field(
            boxsize, num_mesh, power_spectrum,
            bias=bias,
            return_disp=add_RSD,
            seed=field_seed
        )
        sampled_field = smp_field(
            field, mean_density, boxsize, seed=sampling_seed
        )
        position, displacement = pop_field(
            sampled_field, mean_density, boxsize,
            velocity_offset_fields=vec_field,
            seed=drift_seed
        )

        self._size = len(position)
        self._pos = position

        super().__init__(comm=comm)

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("%s generated. ", self.__str__())

        self['Position'] += [boxsize/2] * 3

        if self.attrs['RSD_flag']:
            self._vel_offset = growth_rate * displacement
            if line_of_sight is not None:
                self['Position'] += self['VelocityOffset'] \
                    * np.array(line_of_sight)
            else:
                self['Position'] += self['VelocityOffset'] \
                    * normalise_vector(self['Position'])
            if self.comm is None or self.comm.rank == 0:
                self._logger.info("RSDs added to radial particle positions. ")

    def __str__(self):

        str_root = (
            "LogNormalCatalogue"
            "(nbar={0}, b1={1}, f={2}, boxsize={3}, num_mesh={4}, seed={5})"
        )
        str_args = (
            self.attrs['nbar'],
            self.attrs['bias'],
            self.attrs['growth_rate'],
            self.attrs['BoxSize'],
            self.attrs['Nmesh'],
            self.attrs['seed'],
        )

        return str_root.format(*str_args)

    @column
    def Position(self):
        """Particle positions (in Mpc/:math:`h`).

        """
        return self.make_column(self._pos)

    @column
    def VelocityOffset(self):
        """Appropriately normalised particle velocity offsets (in
        Mpc/:math:`h`) to positions.

        """
        return self.make_column(self._vel_offset)


class GaussianCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum
    with particle velocities predicted by the Zel'dovich approximation.

    Parameters
    ----------
    power_spectrum : callable
        Desired linear matter power spectrum (in cubic Mpc/:math:`h`).
    mean_density : float
        Desired mean particle number density (in cubic :math:`h`/Mpc).
    boxsize : float
        Catalogue box size per dimension (in Mpc/:math:`h`).
    num_mesh : int
        Mesh grid number per dimension for FFT generation.
    bias : float, optional
        Desired particle bias relative to the matter distribution (default
        is 2.).
    add_RSD : bool, optional
        If `True` (default is `False`), add appropriately normalised
        redshift-space velocity offset to particle positions.
    growth_rate : float or None, optional
        Linear growth rate.  Cannot be `None` (default) if `add_RSD` is
        `True` and `Plin` does not have both 'cosmo' and 'redshift'
        attributes.
    line_of_sight : array_like or None, optional
        Line-of-sight direction vector.  If `None` (default), this is set
        to the radial directions.
    seed : int or None, optional
        Random seed of the catalogue (default is `None`).
    comm : :class:`nbodykit.CurrentMPIComm` *or None, optional*
        Current MPI communicator.

    Warnings
    --------
    The field statistics may change after Poisson sampling to discrete
    particles as the density contrast is clipped from below at -1.

    """

    _logger = logging.getLogger("GaussianCatalogue")

    @CurrentMPIComm.enable
    def __init__(self, power_spectrum, mean_density, boxsize, num_mesh,
                 bias=2., add_RSD=False, growth_rate=None, line_of_sight=None,
                 seed=None, comm=None):

        self.comm = comm
        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, _MAX_INT)
            seed = self.comm.bcast(seed)

        self.power_spectrum = power_spectrum
        if hasattr(power_spectrum, 'attrs'):
            self.attrs.update(power_spectrum.attrs)

        self.attrs.update(
            {
                'nbar': mean_density,
                'bias': bias,
                'BoxSize': [boxsize] * 3,
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
        elif growth_rate is None and self.attrs['RSD_flag']:
            raise ValueError(
                "`growth_rate` cannot be None if `add_RSD` is True and "
                "`power_spectrum` does not have 'cosmo' and 'redshift' "
                "attributes. "
            )
        elif not self.attrs['RSD_flag']:
            growth_rate = None
        self.attrs['growth_rate'] = growth_rate

        field_seed, sampling_seed, drift_seed = \
            np.random.RandomState(seed).randint(0, 0xfffffff, size=3)

        field, vec_field = gen_gaussian_field(
            boxsize, num_mesh, power_spectrum,
            bias=bias,
            return_disp=add_RSD,
            seed=field_seed
        )
        sampled_field = smp_field(
            field, mean_density, boxsize, seed=sampling_seed
        )
        position, displacement = pop_field(
            sampled_field, mean_density, boxsize,
            velocity_offset_fields=vec_field,
            seed=drift_seed
        )

        self._size = len(position)
        self._pos = position

        super().__init__(comm=comm)

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("%s generated. ", self.__str__())

        self['Position'] += [boxsize/2] * 3

        if self.attrs['RSD_flag']:
            self._vel_offset = growth_rate * displacement
            if line_of_sight is not None:
                self['Position'] += self['VelocityOffset'] \
                    * np.array(line_of_sight)
            else:
                self['Position'] += self['VelocityOffset'] \
                    * normalise_vector(self['Position'])
            if comm is None or comm.rank == 0:
                self._logger.info("RSDs added to radial particle positions. ")

    def __str__(self):

        str_root = (
            "GaussianCatalogue"
            "(nbar={0}, b1={1}, f={2}, boxsize={3}, Nmesh={4}, seed={5})"
        )
        str_args = (
            self.attrs['nbar'],
            self.attrs['bias'],
            self.attrs['growth_rate'],
            self.attrs['BoxSize'],
            self.attrs['Nmesh'],
            self.attrs['seed'],
        )

        return str_root.format(*str_args)

    @column
    def Position(self):
        """Particle positions (in Mpc/:math:`h`).

        """
        return self.make_column(self._pos)

    @column
    def VelocityOffset(self):
        """Appropriately normalised particle velocity offsets (in
        Mpc/:math:`h`) to positions.

        """
        return self.make_column(self._vel_offset)
