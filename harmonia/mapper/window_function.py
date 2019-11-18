"""
Window function (:mod:`~harmonia.mapper.window`)
===========================================================================

Determine the window function for given survey specifications with
high-density synthetic catalogues.

.. autosummary::

    WindowFunction

|

"""
import warnings

import numpy as np
from mcfit import P2xi
from nbodykit.lab import ConvolvedFFTPower, UniformCatalog, FKPCatalog
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.collections.utils import cartesian_to_spherical as c2s


class WindowFunction:
    """Window function determined from high-density synthetic catalogues.

    Parameters
    ----------
    mask, selection, weight : callable or None, optional
        Survey specification functions (default is `None`).
    source : {'simulation', 'survey'}, optional
        Source type of the catalogue.  If ``'simulation'`` (default),
        `mask`, `selection` and `weight` are assumed to be functions of
        Cartesian coordinates whose origin is the centre of the catalogue;
        if ``'survey'``, `mask`, `selection` and `weight` are assumed to be
        functions of spherical coordinates.

    Attributes
    ----------
    mask : callable or None
        Survey mask.
    selection : callable or None
        Survey selection function.
    weight : callable or None
        Survey weighting.
    synthetic_catalogue : |catalogue_source| or None
        Synthetic catalogue with the specified window.
    power_multipoles : |binned_stats| or None
        Power spectrum multipoles of the window.
    correlation_multipoles : dict or None
        Correlation function multipoles of the window.


    .. |catalogue_source| replace::

        :class:`nbodykit.base.catalog.CatalogSource`


    .. |binned_stats| replace::

        :class:`nbodykit.binned_statistic.BinnedStatistic`

    """

    def __init__(self, mask=None, selection=None, weight=None,
                 source='simulation'):

        self.source, self.mask, self.selection, self.weight = \
            source, mask, selection, weight

        self.synthetic_catalogue = None
        self.power_multipoles = None
        self.correlation_multipoles = None

    def synthethise(self, number_density, boxsize, padding=80.):
        """Synthethise a random catalogue with the specified window and
        additional padding.

        Parameters
        ----------
        number_density : float
            Particle number density (in cubic h/Mpc).
        boxsize : float, array_like
            Catalogue box size (in Mpc/h) as a scalar or a triple of
            scalars.
        padding : float, optional
            Additional box padding needed for accurate window function
            determination (default is 80.).

        Returns
        -------
        :class:`nbodykit.base.catalog.CatalogSource`
            Synthetic catalogue.  Also sets :attr:`synthetic_catalogue`.

        """
        if self.synthetic_catalogue is not None:
            warnings.warn(
                "Synthetic catalogue has already been generated. "
                "It is now being resynthesised."
            )

        catalogue = UniformCatalog(number_density, boxsize)

        catalogue['Location'] = catalogue['Position'] - np.divide(boxsize, 2)

        if self.source == 'simulation':
            pass
        elif self.source == 'survey':
            catalogue['Location'] = c2s(catalogue['Location'])
        else:
            raise ValueError(
                "Unrecognised source type: '{}'. ".format(self.source)
            )

        if callable(self.self.mask):
            catalogue['Weight'] *= self.mask(catalogue['Location'])
        if callable(self.selection):
            catalogue['Weight'] *= self.selection(catalogue['Location'])
        if callable(self.weight):
            catalogue['Weight'] *= self.weight(catalogue['Location'])

        catalogue['NZ'] = number_density * catalogue['Weight']

        self.synthetic_catalogue = FKPCatalog(
            catalogue, None, BoxSize=padding*boxsize
        )

    def power_spectrum_multipoles(self, degrees, kmin=0., kmax=None, dk=None,
                                  **mesh_kwargs):
        """Determine window function power spectrum multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        degrees : int, array_like
            Multipole degrees.
        kmin, kmax, dk : float or None, optional
            Maximum wavenumber, minimum wavenumber or wavenumber bin size
            (default is `None`).
        **mesh_kwargs
            Keyword arguments to be passed to
            :class:`nbodykit.source.mesh.catalog.CatalogMesh` for FFT
            meshgrid painting.

        Returns
        -------
        :class:`nbodykit.binned_statistic.BinnedStatistic`
            Binned power spectrum multipole statistics.  Also sets
            :attr:`power_multipoles`.

        """
        if self.power_multipoles is not None:
            warnings.warn(
                "Power spectrum multipoles have already been computed. "
                "They are now being overwritten."
            )

        if self.synthetic_catalogue is None:
            raise AttributeError(
                "Atribute `synthetic_catalogue` is missing ."
                "Please call `synthesise` method first. "
            )

        if kmax is None:
            kmax = np.pi * mesh_kwargs['Nmesh'] \
                / min(self.synthetic_catalogue.attrs['BoxSize'])
        if dk is None:
            dk = 2 * (2*np.pi / max(self.synthetic_catalogue.attrs['BoxSize']))

        if 'Nmesh' not in mesh_kwargs:
            mesh_kwargs['Nmesh'] = 256
        if 'resampler' not in mesh_kwargs:
            mesh_kwargs['resampler'] = 'tsc'
        if 'compensated' not in mesh_kwargs:
            mesh_kwargs['compensated'] = True

        synthetic_mesh = self.synthetic_catalogue.to_mesh(**mesh_kwargs)

        self.power_multipoles = ConvolvedFFTPower(
            synthetic_mesh, degrees, kmin=kmin, kmax=kmax, dk=dk
        ).poles

        return self.power_multipoles

    def correlator_multipoles(self, degrees, **power_kwargs):
        """Determine window function 2-point correlator multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        degrees : int, array_like
            Multipole degrees.
        **power_kwargs
            Keyword arguments to be passed to
            :class:`~WindowFunction.power_spectrum_multipoles` for
            computing multipoles of the synthetic catalogue.

        Returns
        -------
        dict of {int :code:`:` tuple}
            Hankel transformed separation value and corresponding window
            function correlation multipoles for each of the required
            degrees.  Also sets :attr:`correlation_multipoles`.

        """
        K_MAX = 10.
        K_EXTENSION_PAD = 1e-3
        NUM_K_EXTENSION = 100
        NUM_K_INTERPOL = 10000

        if self.correlation_multipoles is not None:
            warnings.warn(
                "2-point correlator multipoles have already been computed. "
                "They are now being overwritten."
            )

        max_required_degree = max(degrees) + 4
        required_degrees = list(range(0, max_required_degree+1, 2))

        try:
            assert hasattr(
                self.power_multipoles,
                'power_{:d}'.format(max_required_degree)
            )
            power_multipoles = self.power_multipoles
        except AssertionError:
            power_multipoles = self.power_spectrum_multipoles(
                required_degrees, **power_kwargs
            )

        normalisation_amplitude = power_multipoles['power_0'][0]

        k_samples = power_multipoles['k']
        pk_ell_samples = {
            ell: power_multipoles[f'power_{ell}'] for ell in degrees
        }

        k_extended = np.append(
            k_samples,
            np.linspace(
                k_samples[-1] + K_EXTENSION_PAD, K_MAX, num=NUM_K_EXTENSION
            )
        )
        pk_ell_extended = {
            ell: np.append(pk_ell_samples[ell], np.zeros(NUM_K_EXTENSION))
            for ell in degrees
        }

        k_interpol = np.logspace(
            *np.log(k_extended[[0, -1]]), num=NUM_K_INTERPOL
        )
        pk_ell_interpol = {
            Spline(
                k_extended,
                pk_ell_extended[ell] / normalisation_amplitude,
                k=1
            )(k_interpol)
            for ell in degrees
        }

        self.correlation_multipoles = {
            ell: P2xi(k_interpol, l=ell, lowring=True)(pk_ell_interpol[ell])
            for ell in degrees
        }

        return self.correlation_multipoles
