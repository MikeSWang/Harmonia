import logging

import numpy as np
from nbodykit import CurrentMPIComm
from nbodykit.base.catalog import CatalogSource, column

from randomfield import (
    generate_gaussian_randomfields as gen_field,
    perform_lognormal_transformation as trf_field,
    perform_biased_clipping as clp_field,
    perform_poisson_sampling as smp_field
    )
# from harmonia.algorithms import (
#     generate_gaussian_randomfields as gen_field,
#     perform_lognormal_transformation as trf_field,
#     perform_poisson_sampling as smp_field
#     )


class GaussianCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum.

    Attributes
    ----------
    Plin : callable
        Linear matter power spectrum with specified cosmology and redshift.
    comm : :class:`nbodykit.CurrentMPIComm`
        Current MPI communicator.
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
            Linear matter power spectrum with specified cosmology and redshift.
        nbar : float
            Input mean particle number density (lenght unit Mpc/h).
        BoxSize : float, array_like
            Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
        Nmesh : int
            Mesh grid number for FFT generation.
        bias : float, optional
            Particle bias relative to the matter distribution (default is 2.).
        seed : int or None, optional
            Random seed of the catalogue (default is `None`).
        comm : :class:`nbodykit.CurrentMPIComm` or None, optional
            Current MPI communicator.

        """
        self.Plin = Plin
        self.comm = comm

        # Update attributes.
        if not isinstance(BoxSize, list):
            BoxSizes = [BoxSize]*3

        if hasattr(self, 'attrs'):
            self.attrs.update({
                'nbar': nbar,
                'bias': bias,
                'BoxSize': BoxSizes,
                'Nmesh': Nmesh,
                })

        if hasattr(Plin, 'attrs'):
            self.attrs.update(Plin.attrs)

        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        # Generate fields.
        Ncell = nbar * np.prod(BoxSizes) / Nmesh**3

        field_seed, sampling_seed = np.random.RandomState(seed).randint(
            0, 0xfffffff, size=2
            )

        gaussian_field, _ = gen_field(BoxSize, Nmesh, Plin, seed=field_seed)
        clipped_field = clp_field(gaussian_field, bias=bias)
        position = smp_field(clipped_field, Ncell, BoxSize, seed=sampling_seed)

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)


class LogNormalCatalogue(CatalogSource):
    """Log-normal random catalogue of given number density and power spectrum.

    Attributes
    ----------
    Plin : callable
        Linear matter power spectrum with specified cosmology and redshift.
    comm : :class:`nbodykit.CurrentMPIComm`
        Current MPI communicator.
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
            Linear matter power spectrum with specified cosmology and redshift.
        nbar : float
            Input mean particle number density (lenght unit Mpc/h).
        BoxSize : float, array_like
            Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
        Nmesh : int
            Mesh grid number for FFT generation.
        bias : float, optional
            Particle bias relative to the matter distribution (default is 2.).
        seed : int or None, optional
            Random seed of the catalogue (default is `None`).
        comm : :class:`nbodykit.CurrentMPIComm` or None, optional
            Current MPI communicator.

        """
        self.Plin = Plin
        self.comm = comm

        # Update attributes.
        if not isinstance(BoxSize, list):
            BoxSizes = [BoxSize]*3

        if hasattr(self, 'attrs'):
            self.attrs.update({
                'nbar': nbar,
                'bias': bias,
                'BoxSize': BoxSizes,
                'Nmesh': Nmesh,
                })

        if hasattr(Plin, 'attrs'):
            self.attrs.update(Plin.attrs)

        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        # Generate fields.
        Ncell = nbar * np.prod(BoxSizes) / Nmesh**3

        field_seed, sampling_seed = np.random.RandomState(seed).randint(
            0, 0xfffffff, size=2
            )

        gaussian_field, _ = gen_field(BoxSize, Nmesh, Plin, seed=field_seed)
        lognormal_field = trf_field(gaussian_field, bias=bias)
        position = smp_field(
            lognormal_field, Ncell, BoxSize, seed=sampling_seed
            )

        # Initiate the base class.
        self._size = len(position)
        self._pos = position
        super().__init__(comm=comm)

    @column
    def Position(self):
        """Particle positions (in Mpc/h).

        """
        return self.make_column(self._pos)
