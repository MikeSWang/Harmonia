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
from mcfit import P2xi
from nbodykit.lab import ConvolvedFFTPower, FKPCatalog, UniformCatalog
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.collections.utils import cartesian_to_spherical as c2s


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
        self.power_multipoles = None
        self.correlation_multipoles = None

    def synthesise(self, number_density, boxsize, padding=1.):
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
            determination (default is 1.).

        Returns
        -------
        :class:`nbodykit.base.catalog.CatalogSource`
            Synthetic catalogue.  Also sets :attr:`synthetic_catalogue`.

        """
        if self.synthetic_catalogue is not None:
            warnings.warn(
                "Synthetic catalogue already exists. "
                "It is now being resynthesised. "
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
            catalogue['Weight'] *= self.mask(catalogue['Location'])
        if callable(self.selection):
            catalogue['Weight'] *= self.selection(catalogue['Location'])
        if callable(self.weight):
            catalogue['Weight'] *= self.weight(catalogue['Location'])

        catalogue['NZ'] = number_density * catalogue['Weight']

        self.synthetic_catalogue = FKPCatalog(
            catalogue, None, BoxSize=padding*boxsize
        )

    def power_spectrum_multipoles(self, orders, kmin=0., kmax=None, dk=None,
                                  **mesh_kwargs):
        """Determine window function power spectrum multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        orders : int, array_like
            Multipole orders.
        kmin : float, optional
            Minimum wavenumber (default is 0.).
        kmax, dk : float or None, optional
            Maximum wavenumber or wavenumber bin size (default is `None`).
            If `None`, `kmax` is the Nyquist wavenumber determined from
            ``**mesh_kwargs`` and `dk` is the fundamental wavenumber.
        **mesh_kwargs
            Keyword arguments to be passed to
            :class:`nbodykit.source.mesh.catalog.CatalogMesh` for FFT
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

        """
        LOG10_K_MAX = 1.
        NUM_K_EXTENSION = 1000

        if self.synthetic_catalogue is None:
            raise AttributeError(
                "Attribute `synthetic_catalogue` is missing. "
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
            kmax = 0.8 * np.pi * mesh_kwargs['Nmesh'] \
                / min(self.synthetic_catalogue.attrs['BoxSize'])
        if dk is None:
            dk = np.sqrt(3) * 2*np.pi \
                / min(self.synthetic_catalogue.attrs['BoxSize'])

        if self.power_multipoles is not None:
            warnings.warn(
                "Power spectrum multipoles have already been computed. "
                "They are now being overwritten. "
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
            power_multipoles = ConvolvedFFTPower(
                synthetic_mesh, orders, kmin=kmin, kmax=kmax, dk=dk
            ).poles

        valid_bins = (
            ~np.isnan(power_multipoles['modes'])
            & ~np.equal(power_multipoles['modes'], 0)
            & ~(power_multipoles['modes'] % 2)
        ).astype(bool)

        k_samples = power_multipoles['k'][valid_bins]

        self.power_multipoles= {
            'power_{:d}'.format(ell):
                power_multipoles['power_{:d}'.format(ell)][valid_bins].real
            for ell in orders
        }

        normalisation_amplitude = self.power_multipoles['power_0'][0].real
        self.power_multipoles.update(
            {
                var_name: var_vals / normalisation_amplitude
                for var_name, var_vals in self.power_multipoles.items()
                if 'power_' in var_name
            }
        )

        extension_padding = np.mean(np.abs(np.diff(k_samples)))
        extension_first_leg = np.max(k_samples) \
            + extension_padding * np.arange(1, NUM_K_EXTENSION)
        extension_second_leg = np.logspace(
            np.log10(np.max(k_samples) + NUM_K_EXTENSION*extension_padding),
            LOG10_K_MAX,
            num=NUM_K_EXTENSION
        )

        self.power_multipoles['k'] = np.append(
            k_samples, np.append(extension_first_leg, extension_second_leg)
        )
        self.power_multipoles.update(
            {
                var_name: np.append(var_vals, np.zeros(2*NUM_K_EXTENSION-1))
                for var_name, var_vals in self.power_multipoles.items()
                if 'power_' in var_name
            }
        )

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
                "Correlation function multipoles have already been computed. "
                "They are now being overwritten. "
            )

        try:
            assert hasattr(
                self.power_multipoles, 'power_{:d}'.format(max(orders))
            )
            power_multipoles = self.power_multipoles
        except (AttributeError, AssertionError):
            power_multipoles = self.power_spectrum_multipoles(
                orders, **multipoles_kwargs
            )

        k = power_multipoles['k']
        pk_ell = {
            ell: power_multipoles['power_{:d}'.format(ell)]
            for ell in orders
        }

        k_interpol = np.logspace(*np.log10(k[[0, -1]]), num=NUM_INTERPOL)
        pk_ell_interpol = {
            ell: Spline(k, pk_ell[ell], k=1)(k_interpol)
            for ell in orders
        }

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
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

        self.correlation_multipoles = {'s': s_interpol}
        self.correlation_multipoles.update(
            {
                'correlation_{:d}'.format(ell):
                    xi_ell_interpol[ell] / normalisation_amplitude
                for ell in orders
            }
        )

        return self.correlation_multipoles
