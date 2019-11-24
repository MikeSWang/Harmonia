r"""
Cartesian Fourier-space model (:mod:`~harmonia.reader.cartesian_model`)
===========================================================================

Compute Fourier-space :math:`n`-point functions in the Cartesian basis.

.. autosummary::

    WindowedPowerSpectrum
    WindowCorrelation

|

"""
import warnings

import numpy as np
from mcfit import P2xi, xi2P
from nbodykit.lab import cosmology
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.algorithms import CartesianArray
from harmonia.cosmology import scale_dependence_modification


class WindowedPowerSpectrum:
    """Window-convolved power spectrum model for given survey and
    cosmological specifications.

    Parameters
    ----------
    redshift : float, optional
        Current redshift at which the model is evaluated (default is 0.).
    growth_rate : float or None, optional
        Linear growth rate at the current epoch.  If `None` (default), this
        is set to zero and RSD calculations are neglected.
    power_spectrum : callable or None, optional
        Linear matter power spectrum model at the current epoch.
    cosmo : :class:`nbodykit.cosmology.Cosmology` *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating the
        scale-dependent bias.
    window : |window_function|, dict or None, optional
        Window correlation function multipoles (or given at sampled
        separations).

    Attributes
    ----------
    redshift : float
        Current redshift at which the model is evaluated (default is 0.).
    growth_rate : float or None
        Linear growth rate at the current epoch.  If `None`, RSD 
        calculations are neglected.
    power_spectrum : callable or None
        Linear matter power spectrum model at the current epoch.
    window : dict or None
        Window correlation function multipoles.
    cosmo : :class:`nbodykit.cosmology.Cosmology` *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating the
        scale-dependent bias.

    Raises
    ------
    ValueError
        If `power_spectrum` is `None` when `cosmo` is.


    .. |window_function| replace::

        :class:`~harmonia.mapper.window_synthesis.SurveyWindow`

    """

    def __init__(self, redshift=0., growth_rate=None, power_spectrum=None, 
                 cosmo=None, window=None):

        self.redshift = redshift
        self.growth_rate = growth_rate
        self.matter_power_spectrum = power_spectrum

        if cosmo is None:
            if self.matter_power_spectrum is None:
                raise ValueError(
                    "`power_spectrum` cannot be None when `cosmo` is None. "
                )
        else:
            if self.matter_power_spectrum is None:
                self.matter_power_spectrum = cosmology.LinearPower(
                    cosmo, redshift=self.redshift, transfer='CLASS'
                )
            else:
                warnings.warn(
                    "Input `power_spectrum` is used instead of "
                    "the power spectrum associated with `cosmo`. "
                    "Double check their underlying cosmological models "
                    "are consistent. "
                )
            if self.growth_rate is not None:
                cosmo_growth_rate = cosmo.scale_independent_growth_rate(
                    self.redshift
                )
                if not np.isclose(self.growth_rate, cosmo_growth_rate):
                    warnings.warn(
                        "`growth_rate` value inconsistent with `cosmo` model: "
                        "input {}, model predicted value {}. "
                        .format(self.growth_rate, cosmo_growth_rate)
                    )
        self.cosmo = cosmo

        if window is None or isinstance(window, dict):
            self.window = window
        else:
            try:
                assert window.correlation_multipoles is not None
                self.window = window.correlation_multipoles
            except AssertionError:
                default_required_orders = [0, 2, 4, 6, 8]
                self.window = window.correlation_function_multipoles(
                    default_required_orders
                )
                warnings.warn(
                    "Window correlation function multipoles have "
                    "not been computed and are being evaluated now "
                    "at default parameter settings. "
                    "Results may not be accurate. Pass `window` "
                    "with attribute `correlation_multipoles` computed "
                    "at pre-configured parameter settings "
                    "for more reliable results. "
                )

        self._wavenumbers = None

    @property
    def wavenumbers(self):
        """Wavenumbers at which the windowed model is evaluated.

        Returns
        -------
        float, array_like

        """
        return self._wavenumbers

    @wavenumbers.setter
    def wavenumbers(self, values):
        self._wavenumbers = values

    def convolved_multipoles(self, orders, b_1, f_nl=None, nbar=None,
                             wavenumbers=None, tracer_parameter=1.,
                             contrast=np.inf):
        """Compute the convolved model of power spectrum multipoles.

        Parameters
        ----------
        orders : list or tuple of int
            Order(s) of the multipole, in ``{0, 2, 4}``, sorted in
            ascending order.
        b_1 : float
            Scale-independent linear bias.
        f_nl : float or None, optional
            Local primordial non-Gaussianity parameter (default is `None`).
        nbar : float or None, optional
            Mean particle number density (in cubic h/Mpc).  If `None`
            (default), shot noise is neglected.
        wavenumbers : float, array_like or None, optional
            Wavenumbers at which the model is evaluated (default is
            `None`).  If `None`, :attr:`wavenumbers` property must be set;
            if not `None`, this overwrites the existing :attr:`wavenumbers`
            property.
        tracer_parameter : float, optional
            Tracer species--dependent parameter for bias modulation
            (default is 1.).
        contrast : float, optional
            Effectively downscale `nbar` (if not `None`) by 
            ``1 + 1/constrast``.  Default is ``numpy.inf``.

        Returns
        -------
        dict
            Convolved power spectrum multipoles evaluated at the given
            wavenumbers.

        Raises
        ------
        ValueError
            If `orders` contains an element not theoretically predicted,
            i.e. not in ``{0, 2, 4}``.
        ValueError
            If `wavenumbers` is `None` when :attr:`wavenumbers` property is
            also `None`.

        """
        LOG_K_INTERPOL_RANGE = -5, 1
        NUM_INTERPOL = 10000
        PREDICTED_ORDERS = [0, 2, 4]

        if not set(orders).issubset(set(PREDICTED_ORDERS)):
            raise ValueError("`orders` must be a subset of {0, 2, 4}. ")

        if wavenumbers is None:
            if self.wavenumbers is None:
                raise ValueError(
                    "`wavenumbers` cannot be None when the correponding "
                    "property of the class instance is None. "
                )
            wavenumbers = self.wavenumbers

        if f_nl is not None and self.cosmo is None:
            raise AttributeError(
                "Attribute `cosmo` cannot be None "
                "when scale-dependence is introduced by non-null `f_nl`. "
            )

        if self.window is None:
            if f_nl is None:
                b_k = b_1 * np.ones_like(wavenumbers)
            else:
                b_k = b_1 + f_nl * (b_1 - tracer_parameter) \
                    * scale_dependence_modification(self.cosmo, self.redshift)(
                        wavenumbers
                    )

            pk_ell_convolved = {
                "power_{}".format(ell):
                    self.kaiser_factors(ell, b_k, self.growth_rate) \
                    * self.matter_power_spectrum(wavenumbers)
                for ell in orders
            }

            pk_ell_convolved['k'] = wavenumbers

            return pk_ell_convolved

        s = self.window['s']

        k_interpol = np.logspace(*LOG_K_INTERPOL_RANGE, num=NUM_INTERPOL)

        if f_nl is None:
            b_k_interpol = b_1 * np.ones_like(k_interpol)
        else:
            b_k_interpol = b_1 + f_nl * (b_1 - tracer_parameter) \
                * scale_dependence_modification(self.cosmo, self.redshift)(
                    k_interpol
                )

        pk_ell_interpol = {
            ell: self.kaiser_factors(ell, b_k_interpol, self.growth_rate) \
                * self.matter_power_spectrum(k_interpol)
            for ell in PREDICTED_ORDERS
        }

        xi_ell = {
            ell: Spline(
                *P2xi(k_interpol, l=ell, lowring=True)(
                    pk_ell_interpol[ell], extrap=False
                ),
                k=1
            )(s)
            for ell in PREDICTED_ORDERS
        }

        xi_ell_convolved = {}
        if 0 in orders:
            xi_ell_convolved[0] = \
                xi_ell[0] * self.window['correlation_0'] \
                + 1/5 * xi_ell[2] * self.window['correlation_2'] \
                + 1/9 * xi_ell[4] * self.window['correlation_4']
        if 2 in orders:
            xi_ell_convolved[2] = \
                xi_ell[0] * self.window['correlation_2'] \
                + xi_ell[2] * (
                    self.window['correlation_0']
                    + 2/7 * self.window['correlation_2']
                    + 2/7 * self.window['correlation_4']
                ) \
                + xi_ell[4] * (
                    2/7 * self.window['correlation_2']
                    + 100/693 * self.window['correlation_4']
                    + 25/143 * self.window['correlation_6']
                )
        if 4 in orders:
            xi_ell_convolved[4] = \
                xi_ell[0] * self.window['correlation_4'] \
                + xi_ell[2] * (
                    18/35 * self.window['correlation_2']
                    + 20/77 * self.window['correlation_4']
                    + 45/143 * self.window['correlation_6']
                ) \
                + xi_ell[4] * (
                    self.window['correlation_0']
                    + 20/77 * self.window['correlation_2']
                    + 162/1001 * self.window['correlation_4']
                    + 20/143 * self.window['correlation_6']
                    + 490/2431 * self.window['correlation_8']
                )

        pk_ell_convolved_sampled = {
            ell: xi2P(s, l=ell, lowring=True)(
                xi_ell_convolved[ell], extrap=False
            )
            for ell in orders
        }

        pk_ell_convolved = {
            "power_{}".format(ell):
                Spline(*pk_ell_convolved_sampled[ell], k=1)(wavenumbers)
            for ell in orders
        }

        pk_ell_convolved['k'] = wavenumbers

        if nbar is not None and 0 in orders:
            pk_ell_convolved["power_0"] += (1 + 1/contrast) / nbar \
                * np.ones_like(wavenumbers)

        return pk_ell_convolved

    @staticmethod
    def kaiser_factors(order, bias, growth_rate):
        """Return the standard Kaiser power spectrum multipole as a
        multipole of the matter power spectrum.

        Notes
        -----
        `bias` and `growth_rate` must be specified at the same redshift.

        Parameters
        ----------
        order : int
            Order of the multipole, ``order >= 0``.
        bias : float, array_like
            Scale-independent linear bias of the tracer particles.
        growth_rate : float or None
            Linear growth rate.

        Returns
        -------
        kaiser_factor : float, array_like
            Standard Kaiser power spectrum multipole factor.

        """
        ell, b = order, np.atleast_1d(bias)
        if growth_rate is None:
            f = 0.
        else:
            f = growth_rate

        if ell == 0:
            kaiser_factor = b**2 + 2/3 * b * f + 1/5 * f**2
        elif ell == 2:
            kaiser_factor = 4/3 * b * f + 4/7 * f**2
        elif ell == 4:
            kaiser_factor = 8/35 * f**2
        else:
            kaiser_factor = np.zeros_like(b)

        return np.squeeze(kaiser_factor)


class WindowCorrelation:
    """Window-induced correlation matrix for power spectrum multipoles at
    given wavenumbers.

    Parameters
    ----------
    fiducial_windowed_power_model : dict
        The fiducial windowed power spectrum model based at which the
        window-induced correlation is estimated, with the mandatory
        key ``'k'`` for the wavenumbers corresponding to the correlation
        matrix components.
    pivot : {'orders', 'scale'}
        The order in which the correlation matrix components are arranged:
        if ``'orders'``, the compoenents are in ascending multipole order 
        and then in ascending order of the wavenumber; if ``'scale'``, the 
        components are similarly ordered by the wavenumber first and then 
        the multipole order.

    Attributes
    ----------
    pivot : {'orders', 'scale'}
        The same as the `pivot` argument.
    orders : list of int
        Order of the multipoles.
    wavenumbers : float :class:`numpy.ndarray`
        Wavenumbers to which the window correlation components correspond.
    fiducial_power_multipoles : float :class:`numpy.ndarray`
        Power multipoles to which the window correlation components
        correspond and at which the window correlation is estimated.

    """

    def __init__(self, fiducial_windowed_power_model, pivot):

        fiducial_model_array = CartesianArray(
            fiducial_windowed_power_model, "k", "power_"
        )

        self.pivot = pivot
        self.orders = list(map(int, fiducial_model_array.sorted_vars))

        if self.pivot == 'orders':
            self.fiducial_diagonal_multipoles, self.wavenumbers = \
                fiducial_model_array.unfold('variable')
        elif self.pivot == 'scale':
            self.fiducial_diagonal_multipoles, self.wavenumbers = \
                fiducial_model_array.unfold('coord')

        self._window_correlation = None

    @property
    def window_correlation(self):
        """Covariance matrix induced by the window at
        :attr:`fiducial_power_multipoles`.

        Notes
        -----
        The ordering of the components in each dimension must be consistent
        with :attr:`pivot`.  The components must be consistent with
        :attr:`wavenumbers`.

        Raises
        ------
        ValueError
            If set to a matrix of inconsistent dimensions with the
            :attr:`wavenumbers` property.

        """
        return self._window_correlation

    @window_correlation.setter
    def window_correlation(self, covar_estimate):

        covar_estimate = np.squeeze(np.array(covar_estimate))
        if covar_estimate.ndim != 2 \
                or covar_estimate.shape != (len(self.wavenumbers),) * 2:
            raise ValueError(
                "The value of the `window_correlation` property "
                "must be a matrix of dimensions consistent with the "
                "`wavenumbers` property. "
            )

        self._window_correlation = covar_estimate
