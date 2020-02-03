"""
Window synthesis (:mod:`~harmonia.mapper.window_synthesis`)
===========================================================================

Determine the window function for given survey specifications with
high-density synthetic catalogues.

.. autosummary::

    SurveyWindow

|

"""
import warnings

import numpy as np
import psutil
from mcfit import P2xi
from nbodykit.lab import FKPCatalog, UniformCatalog
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.collections.utils import cartesian_to_spherical as c2s
from harmonia.mapper.cartesian_reduction import CartesianMap


class SurveyWindow:
    """Survey window determined from high-density synthetic catalogues.

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
    power_multipoles : dict or None
        Power spectrum multipoles of the window.
    correlation_multipoles : dict or None
        Correlation function multipoles of the window at sampled
        separations.


    .. |catalogue_source| replace::

        :class:`nbodykit.base.catalog.CatalogSource`

    """

    def __init__(self, mask=None, selection=None, weight=None,
                 source='simulation'):

        self.source, self.mask, self.selection, self.weight = \
            source, mask, selection, weight

        self.synthetic_catalogue = None
        self._raw_power_multipoles = None

        self.power_multipoles = None
        self.correlation_multipoles = None

    def synthesise(self, number_density, boxsize, padding=1.):
        """Synthesise a random catalogue with the specified window and
        additional padding.

        Parameters
        ----------
        number_density : float
            Particle number density (in cubic :math:`h`/Mpc).
        boxsize : float, array_like
            Catalogue box size (in Mpc/:math:`h`) as a scalar or a triple
            of scalars.
        padding : float, optional
            Additional box padding needed for accurate window function
            determination (default is 1.).

        Returns
        -------
        :class:`nbodykit.base.catalog.CatalogSource`
            Synthetic catalogue.  Also sets :attr:`synthetic_catalogue`.

        """
        if self.synthetic_catalogue is not None:
            warnings.warn(
                "Synthetic catalogue already exists; "
                "it is now being resynthesised. "
            )

        catalogue = UniformCatalog(number_density, boxsize)

        catalogue['Location'] = catalogue['Position'] - np.divide(boxsize, 2)

        if self.source == 'simulation':
            pass
        elif self.source == 'survey':
            catalogue['Location'] = c2s(catalogue['Location'])
        else:
            raise ValueError(f"Unrecognised source type: {self.source}. ")

        if callable(self.mask):
            catalogue['Selection'] *= self.mask(catalogue['Location'])
        if callable(self.selection):
            catalogue['Weight'] *= self.selection(catalogue['Location'])
        if callable(self.weight):
            catalogue['Weight'] *= self.weight(catalogue['Location'])

        catalogue['NZ'] = number_density * catalogue['Weight']

        self.synthetic_catalogue = FKPCatalog(
            catalogue, None, BoxSize=padding*boxsize
        )

    def power_spectrum_multipoles(self, orders, kmin=1.e-6, kmax=None, dk=None,
                                  **mesh_kwargs):
        """Determine window function power spectrum multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        orders : int, array_like
            Multipole orders.
        kmin : float, optional
            Minimum wavenumber (default is 1.e-6).
        kmax, dk : float or None, optional
            Maximum wavenumber or wavenumber bin size (default is `None`).
            If `None`, `kmax` is the Nyquist wavenumber determined from
            ``**mesh_kwargs`` and `dk` is the fundamental wavenumber.
        **mesh_kwargs
            Keyword arguments to be passed to |map_multipoles| for FFT
            mesh conversion.  If not given, default settings are used
            (512 meshes per dimension and triangular-shaped cloud ``'tsc'``
            interpolation with compensation and interlacing).

        Returns
        -------
        dict
            Binned power spectrum multipole statistics holding variables
            ``'k'`` for the bin wavenumber and binned multipole values
            ``'power_0'`` etc.  Also sets :attr:`power_multipoles`.

        Raises
        ------
        AttributeError
            If :attr:`synthetic_catalogue` is missing and
            :meth:`~SurveyWindow.synthesise` needs to be called first.


        .. |map_multipoles| replace::

            :meth:`~.cartesian_reduction.CartesianMap.power_multipoles`

        """
        LOG10_K_MAX = 1.
        NUM_K_EXTENSION = 1000
        NUM_INTERPOL_DISPLAY = pow(2, 10)

        if self.synthetic_catalogue is None:
            raise AttributeError(
                "Attribute `synthetic_catalogue` is missing. "
                "Please call the `synthesise` method first. "
            )

        if 'num_mesh' not in mesh_kwargs:
            if psutil.virtual_memory().available > 10 * 768**3:
                mesh_kwargs['num_mesh'] = 768
            else:
                mesh_kwargs['num_mesh'] = 512

        cartesian_map = CartesianMap(self.synthetic_catalogue, **mesh_kwargs)

        if kmax is None:
            kmax = np.pi * mesh_kwargs['num_mesh'] \
                / min(self.synthetic_catalogue.attrs['BoxSize'])
        if dk is None:
            dk = np.sqrt(3) * 2*np.pi \
                / min(self.synthetic_catalogue.attrs['BoxSize'])

        if self._raw_power_multipoles is not None:
            warnings.warn(
                "Power spectrum multipoles have already been computed; "
                "they are now being overwritten. "
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=RuntimeWarning,
                message=".*invalid value encountered.*"
            )
            warnings.filterwarnings(
                'ignore',
                category=RuntimeWarning,
                message="divide by zero encountered in double_scalars"
            )
            measured_power_multipoles = cartesian_map.power_multipoles(
                orders, kmin=kmin, kmax=kmax, dk=dk
            )

        normalisation_amplitude = measured_power_multipoles['power_0'][0].real

        self._raw_power_multipoles = {
            ell: measured_power_multipoles['power_{:d}'.format(ell)]
                 / normalisation_amplitude
            for ell in orders
        }

        # Linear padding and then logarithmic padding.
        k_measured = measured_power_multipoles['k']
        extension_padding = np.mean(np.abs(np.diff(k_measured)))
        extension_first_leg = np.max(k_measured) \
            + extension_padding * np.arange(1, NUM_K_EXTENSION)
        extension_second_leg = np.logspace(
            np.log10(np.max(k_measured) + extension_padding*NUM_K_EXTENSION),
            LOG10_K_MAX,
            num=NUM_K_EXTENSION
        )
        k_extension = np.append(extension_first_leg, extension_second_leg)

        self._raw_power_multipoles.update(
            {'k': np.append(k_measured, k_extension)}
        )
        self._raw_power_multipoles.update(
            {
                ell: np.append(
                    self._raw_power_multipoles[ell],
                    np.zeros(2*NUM_K_EXTENSION-1)
                )
                for ell in orders
            }
        )

        k_samples = np.logspace(
            *np.log10(self._raw_power_multipoles['k'][[0, -1]]),
            num=NUM_INTERPOL_DISPLAY
        )
        self.power_multipoles = {
            'power_{:d}'.format(ell): Spline(
                self._raw_power_multipoles['k'],
                self._raw_power_multipoles[ell],
                k=1
            )(k_samples)
            for ell in orders
        }
        self.power_multipoles.update({'k': k_samples})

        return self.power_multipoles

    def correlation_function_multipoles(self, orders, **multipoles_kwargs):
        """Determine window function 2-point correlator multipoles from the
        synthetic catalogue by Hankel transform of the power spectrum
        multipoles.

        Parameters
        ----------
        orders : int, array_like
            Multipole orders.
        **multipoles_kwargs
            Keyword arguments to be passed to
            :meth:`~.SurveyWindow.power_spectrum_multipoles` for
            computing multipoles of the synthetic catalogue.

        Returns
        -------
        dict
            Hankel transformed window correlation function multipoles for
            each of the orders at sampled separation values.  Also sets
            :attr:`correlation_multipoles`.

        """
        NUM_INTERPOL = pow(2, 14)

        if self.correlation_multipoles is not None:
            warnings.warn(
                "Correlation function multipoles have already been computed; "
                "they are now being overwritten. "
            )

        if self._raw_power_multipoles is None \
                or max(orders) not in self._raw_power_multipoles:
            _ = self.power_spectrum_multipoles(orders, **multipoles_kwargs)

        k = self._raw_power_multipoles['k']
        k_interpol = np.logspace(*np.log10(k[[0, -1]]), num=NUM_INTERPOL)

        pk_ell_interpol = {
            ell: Spline(k, self._raw_power_multipoles[ell], k=1)(k_interpol)
            for ell in orders
        }

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=(
                    "The default value of extrap has been changed to False, "
                    "set it to True if you cannot reproduce previous results"
                )
            )
            xi_ell = {
                ell: P2xi(k_interpol, l=ell, lowring=True)(
                    pk_ell_interpol[ell], extrap=False
                )
                for ell in orders
            }

        normalisation_amplitude = xi_ell[0][-1][0]

        s_interpol = np.logspace(
            np.log10(max([xi_ell[ell][0][0] for ell in orders])),
            np.log10(min([xi_ell[ell][0][-1] for ell in orders])),
            num=NUM_INTERPOL
        )
        xi_ell_interpol = {
            ell: Spline(*xi_ell[ell], k=1)(s_interpol)
            for ell in orders
        }

        self.correlation_multipoles = {
            'correlation_{:d}'.format(ell):
                xi_ell_interpol[ell] / normalisation_amplitude
            for ell in orders
        }
        self.correlation_multipoles.update({'s': s_interpol})

        return self.correlation_multipoles
