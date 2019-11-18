"""
Window synthesis (:mod:`~harmonia.mapper.window_synthesis`)
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
        Survey mask, selection or weight function (default is `None`).
    source : {'simulation', 'survey'}, optional
        Catalogue source type.  If ``'simulation'`` (default), then `mask`,
        `selection` and `weight` are assumed to be functions of Cartesian
        coordinates whose origin is the centre of the catalogue; if
        ``'survey'``, then `mask`, `selection` and `weight` are assumed to
        be functions of spherical coordinates.

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
    correlation_multipoles : dict of {int: tuple} or None
        Correlation function multipoles of the window at sampled
        separations.


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

    def synthesise(self, number_density, boxsize, padding=80.):
        """Synthesise a random catalogue with the specified window and
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

        if callable(self.mask):
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
        kmin : float, optional
            Minimum wavenumber (default is 0.).
        kmax, dk : float or None, optional
            Maximum wavenumber or wavenumber bin size (default is `None`).
            If `None`, `kmax` is bounded above the Nyquist wavenumber and
            `dk` is twice the fundamental wavenumber.
        **mesh_kwargs
            Keyword arguments to be passed to
            :class:`nbodykit.source.mesh.catalog.CatalogMesh` for FFT
            meshgrid painting.  If not given, default settings are used
            (512 meshes per dimension and triangular-shaped cloud
            interpolation with compensation and interlacing).

        Returns
        -------
        :class:`nbodykit.binned_statistic.BinnedStatistic`
            Binned power spectrum multipole statistics holding variables
            ``'modes'`` and ``'k'`` for the number of modes and the
            average wavenumber in each bin, as well as binned multipole
            values ``'power_0'`` etc.  Also sets :attr:`power_multipoles`.

        Raises
        ------
        AttributeError
            If :attr:`synthetic_catalogue` is missing and
            :meth:`~WindowFunction.synthesise` needs to be called first.

        """
        if self.synthetic_catalogue is None:
            raise AttributeError(
                "Attribute `synthetic_catalogue` is missing ."
                "Please call the `synthesise` method first. "
            )

        if 'Nmesh' not in mesh_kwargs:
            mesh_kwargs['Nmesh'] = 512
        if 'resampler' not in mesh_kwargs:
            mesh_kwargs['resampler'] = 'tsc'
        if 'compensated' not in mesh_kwargs:
            mesh_kwargs['compensated'] = True
        if 'interlaced' not in mesh_kwargs:
            mesh_kwargs['interlaced'] = True

        synthetic_mesh = self.synthetic_catalogue.to_mesh(**mesh_kwargs)

        if kmax is None:
            kmax = np.pi * mesh_kwargs['Nmesh'] \
                / min(self.synthetic_catalogue.attrs['BoxSize'])
        if dk is None:
            dk = 2*(2*np.pi / max(self.synthetic_catalogue.attrs['BoxSize']))

        if self.power_multipoles is not None:
            warnings.warn(
                "Power spectrum multipoles have already been computed. "
                "They are now being overwritten. "
            )

        self.power_multipoles = ConvolvedFFTPower(
            synthetic_mesh, degrees, kmin=kmin, kmax=kmax, dk=dk
        ).poles

        normalisation_amplitude = self.power_multipoles['power_0'][0].real

        self.power_multipoles.update({
            ell: self.power_multipoles['power_{:d}'.format(ell)].real \
                / normalisation_amplitude
            for ell in degrees
        })

        return self.power_multipoles

    def correlation_function_multipoles(self, degrees, **multipoles_kwargs):
        """Determine window function 2-point correlator multipoles from the
        synthetic catalogue by Hankel transform of the power spectrum
        multipoles.

        Parameters
        ----------
        degrees : int, array_like
            Multipole degrees.
        **multipoles_kwargs
            Keyword arguments to be passed to
            :meth:`~WindowFunction.power_spectrum_multipoles` for
            computing multipoles of the synthetic catalogue.

        Returns
        -------
        dict of {int :code:`:` tuple}
            Hankel transformed pairs of separation values and corresponding
            window correlation function multipoles for each of the degrees.
            Also sets :attr:`correlation_multipoles`.

        """
        K_MAX = 10.
        NUM_K_EXTENSION = 100
        NUM_INTERPOL = 10000

        if self.correlation_multipoles is not None:
            warnings.warn(
                "2-point correlator multipoles have already been computed. "
                "They are now being overwritten."
            )

        try:
            assert 'power_{:d}'.format(max(degrees)) \
                in self.power_multipoles.variables
            power_multipoles = self.power_multipoles
        except (AttributeError, AssertionError):
            power_multipoles = self.power_spectrum_multipoles(
                degrees, **multipoles_kwargs
            )

        bin_cleansing = ~np.isnan(power_multipoles['k'])
        extension_padding = np.mean(np.diff(power_multipoles['k']))

        k_samples = power_multipoles['k'][bin_cleansing]
        pk_ell_samples = {
            ell: power_multipoles[f'power_{ell}'][bin_cleansing].real
            for ell in degrees
        }

        k_extended = np.append(
            k_samples,
            np.linspace(
                k_samples[-1] + extension_padding,
                K_MAX,
                num=NUM_K_EXTENSION
            )
        )
        pk_ell_extended = {
            ell: np.append(pk_ell_samples[ell], np.zeros(NUM_K_EXTENSION))
            for ell in degrees
        }

        k_interpol = np.logspace(
            *np.log10(k_extended[[0, -1]]),
            num=NUM_INTERPOL
        )
        pk_ell_interpol = {
            ell: Spline(k_extended, pk_ell_extended[ell], k=1)(k_interpol)
            for ell in degrees
        }

        xi_ell = {
            ell: P2xi(k_interpol, l=ell, lowring=True)(pk_ell_interpol[ell])
            for ell in degrees
        }

        normalisation_amplitude = xi_ell[0][1][0]

        r_interpol = np.logspace(
            np.log10(max([xi_ell[ell][0][0] for ell in degrees])),
            np.log10(min([xi_ell[ell][0][-1] for ell in degrees])),
            num=NUM_INTERPOL
        )
        xi_ell_interpol = {
            ell: Spline(*xi_ell[ell], k=1)(r_interpol)
            for ell in degrees
        }

        self.correlation_multipoles = {}
        self.correlation_multipoles['r'] = r_interpol
        self.correlation_multipoles.update({
            'correlation_{:d}'.format(ell): \
                xi_ell_interpol[ell] / normalisation_amplitude
            for ell in degrees
        })

        return self.correlation_multipoles
