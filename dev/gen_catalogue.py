import logging

import numpy as np
from nbodykit.base.catalog import CatalogSource, column
from nbodykit import CurrentMPIComm

from gaussian_random_field import gen_gaussian_random_field as gen_grf
# from harmonia.algorithms import gen_gaussian_random_field as gen_grf


class GaussianRandomCatalogue(CatalogSource):
    """Gaussian random catalogue of given number density and power spectrum.

    Attributes
    ----------

    """

    _logger = logging.getLogger("GaussianRandomCatalogue")

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
        self.attrs = {
            'nbar': nbar,
            'bias': bias,
            'BoxSize': BoxSize,
            'Nmesh': Nmesh,
            }

        if hasattr(Plin, 'attrs'):
            self.attrs.update(Plin.attrs)

        if seed is None:
            if self.comm.rank == 0:
                seed = np.random.randint(0, 4294967295)
            seed = self.comm.bcast(seed)
        self.attrs['seed'] = seed

        grf = gen_grf(BoxSize, Nmesh, Plin, seed=seed)



