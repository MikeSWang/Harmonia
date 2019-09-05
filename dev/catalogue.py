import logging

import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.base.catalog import CatalogSource, column

from random_field import (
    generate_gaussian_random_fields as gen_field,
    perform_biased_clipping as clp_field,
    perform_lognormal_transformation as trf_field,
    perform_poisson_sampling as smp_field,
    perform_particle_population as pop_prtcl,
    )
# from harmonia.algorithms import (
#     generate_gaussian_random_fields as gen_field,
#     perform_biased_clipping as clp_field,
#     perform_lognormal_transformation as trf_field,
#     perform_poisson_sampling as smp_field
#     perform_particle_population as pop_prtcl,
#     )


class GaussianCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum.

    Attributes
    ----------
    comm : :class:`nbodykit.CurrentMPIComm`
        Current MPI communicator.
    Plin : callable
        Linear matter power spectrum with specified cosmology and redshift.
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
            Linear matter power spectrum (in cubic Mpc/h) with specified
            cosmology and redshift.
        nbar : float
            Input mean particle number density (in cubic h/Mpc).
        BoxSize : float
            Catalogue box size (in Mpc/h) as a scalar.
        Nmesh : int
            Mesh grid number for discrete generation.
        bias : float, optional
            Particle bias relative to the matter distribution (default is 2.).
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

        gaussian_field, _ = gen_field(BoxSize, Nmesh, Plin, seed=field_seed)
        biased_field = clp_field(gaussian_field, bias=bias)
        sampled_field = smp_field(
            biased_field, nbar, BoxSize, seed=sampling_seed
            )

        position = pop_prtcl(sampled_field, nbar, BoxSize, seed=drift_seed)

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
        Linear matter power spectrum with specified cosmology and redshift.
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
            Linear matter power spectrum (in cubic Mpc/h) with specified
            cosmology and redshift.
        nbar : float
            Input mean particle number density (in cubic h/Mpc).
        BoxSize : float
            Catalogue box size (in Mpc/h) as a scalar.
        Nmesh : int
            Mesh grid number for discrete generation.
        bias : float, optional
            Particle bias relative to the matter distribution (default is 2.).
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

        gaussian_field, _ = gen_field(BoxSize, Nmesh, Plin, seed=field_seed)
        biased_field = trf_field(gaussian_field, bias=bias)
        sampled_field = smp_field(
            biased_field, nbar, BoxSize, seed=sampling_seed
            )
        position = pop_prtcl(
            sampled_field, nbar, BoxSize, seed=drift_seed
            )

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
