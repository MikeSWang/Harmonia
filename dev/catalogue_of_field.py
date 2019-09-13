import logging

import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.base.catalog import CatalogSource, column

import fieldrc
from harmonia.algorithms import (
    generate_gaussian_random_field as gen_gaussian_field,
    generate_lognormal_random_field as gen_lognormal_field,
    poisson_sample as smp_field,
    populate_particles as pop_field,
    )


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
