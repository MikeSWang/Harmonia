"""
Cartesian Fourier reduction (:mod:`~harmonia.mapper.cartesian_reduction`)
===========================================================================

Reduce discrete catalogues to compressed Fourier-space maps in Cartesian
coordinates.

.. autosummary::

    CartesianMap

|

"""
import logging

import numpy as np
from nbodykit.lab import ConvolvedFFTPower


class CartesianMap:
    """Compressed Cartesian Fourier-space map.

    Parameters
    ----------
    catalogue_pair : |FKP_catalogue|
        A FKP pair of catalogues consisting of the data and the random with
        the same survey window.
    num_mesh : int, optional
        Mesh number per dimension to interpolate the discrete catalogues.
    resampler : {'cic', 'tsc', 'pcs'}, optional
        Grid assignment resampler (default is ``'tsc'``) for catalogue
        interpolation.
    interlacing : bool, optional
        If `True` (default), use mesh interlacing for aliasing mitigation.

    Attributes
    ----------
    mesh : :class:`nbodykit.base.mesh.MeshSource`
        FFT mesh of the interpolated catalogues.


    .. |FKP_catalogue| replace::

        :class:`nbodykit.algorithms.convpower.catalog.FKPCatalog`

    """

    _logger = logging.getLogger('CartesianMap')

    def __init__(self, catalogue_pair, num_mesh=256, resampler='tsc',
                 interlacing=True):

        self.mesh = catalogue_pair.to_mesh(
            Nmesh=num_mesh,
            compensated=True,
            resampler=resampler,
            interlaced=interlacing
        )
        self._logger.debug("FKP catalogue painted to mesh. ")

    def power_multipoles(self, orders, kmin=1.e-4, kmax=None, dk=None):
        """Compress the FFT mesh into power spectrum multipoles.

        Parameters
        ----------
        orders : list of int
            Orders of the power multipoles.
        kmin : float, optional
            The wavenumber minimum to be used (default is 1.e-4).
        kmin, kmax, dk : float or None, optional
            The wavenumber maximum or bin size to be used (default is
            `None`).

        Returns
        -------
        multipoles : dict
            Power multipoles of specified orders at binned wavenumbers
            (given by the key ``'k'``) together with the mode count in
            each bin (given by the key ``'Nk'``).

        """
        power = ConvolvedFFTPower(
            self.mesh, poles=orders, kmin=kmin, kmax=kmax, dk=dk
        ).poles

        valid_bins = ~np.equal(power['modes'], 0) \
            & ~np.equal(power['modes'], 1)

        multipoles = {
            var_name: power[var_name][valid_bins].real
            for var_name in power if 'power_' in var_name
        }
        multipoles['k'] = power['k'][valid_bins]
        multipoles['Nk'] = power['modes'][valid_bins]

        self._logger.debug(
            "Power spectrum multipoles computed for orders %s. ", str(orders)
        )

        return multipoles
