r"""
Cartesian Fourier-space model (:mod:`~harmonia.reader.cartesian_model`)
===========================================================================

Compute Fourier-space :math:`n`-point functions in the Cartesian basis.

"""
import warnings

import numpy as np
from mcfit import P2xi, xi2P
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.cosmology import scale_dependence_modification


class PowerSpectrumConvolution:
    """Compute the window-convolved power spectrum model for given survey
    and cosmological specifications.

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
        Linear growth rate at the current epoch.  If `None` (default), this
        is set to zero and RSD calculations are neglected.
    power_spectrum : callable or None
        Linear matter power spectrum model at the current epoch.
    window : dict or None
        Window correlation function multipoles.
    cosmo : :class:`nbodykit.cosmology.Cosmology` *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating the
        scale-dependent bias.


    .. |window_function| replace::

        :class:`~harmonia.mapper.window_synthesis.WindowFunction`

    """

    def __init__(self, redshift=0., growth_rate=None, power_spectrum=None,
                 cosmo=None, window=None):

        self.redshift = redshift
        self.growth_rate = growth_rate
        self.matter_power_spectrum = power_spectrum

        if cosmo is not None:
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
                cosmo_growth_rate = \
                    cosmo.scale_independent_growth_rate(self.redshift)
                if not np.isclose(self.growth_rate, cosmo_growth_rate):
                    warnings.warn(
                        "`growth_rate` value inconsistent with `cosmo` model: "
                        "input {}, model predicted value {}. "
                        .format(self.growth_rate, cosmo_growth_rate)
                    )
        self.cosmo = cosmo

        if window is not None:
            try:
                assert window.correlation_multipoles is not None
                self.window = window.correlation_multipoles
            except AssertionError:
                self.window = window.correlation_function_multipoles([0, 2, 4])
                warnings.warn(
                    "Window correlation function multipoles have not been "
                    "computed and are being evaluated now at default "
                    "parameter settings. Results may not be accurate. "
                    "Pass `window` with attribute `correlation_multipoles` "
                    "comptued at pre-configured parameter settings for more"
                    "reliable results. "
                )
        else:
            self.window = None

    def convolved_multipoles(self, wavenumbers, degrees, b_1, f_nl=None,
                             nbar=None, tracer_parameter=1., contrast=np.inf):
        """Compute the convolved model of power spectrum multipoles.

        Parameters
        ----------
        wavenumbers : float, array_like
            Wavenumbers at which the model is evaluated.
        degrees : list or tuple of int
            Order of the multipole, ``degrees >= 0``.  This is sorted in
            ascending order.
        b_1 : float
            Scale-independent linear bias.
        f_nl : float or None, optional
            Local primordial non-Gaussianity parameter (default is `None`).
        nbar : float or None, optional
            Mean particle number density (in cubic h/Mpc).  If `None`
            (default), shot noise is neglected.
        tracer_parameter : float, optional
            Tracer species--dependent parameter for bias modulation
            (default is 1.).
        contrast : float, optional
            Downscale `nbar` (if not `None`) by ``1 + 1/constrast``.
            Default is ``numpy.inf``.

        Returns
        -------
        convolved_multipoles : (list of) float :class:`numpy.ndarray`
            Convolved power spectrum multipoles evaluated at the given
            wavenumbers.

        """
        LOG_K_INTERPOL_RANGE = -5, 1
        NUM_INTERPOL = 10000

        if self.cosmo is None:
            if f_nl is not None:
                raise AttributeError(
                    "Attribute `cosmo` cannot be None "
                    "when scale-dependence is introduced by non-null `f_nl`. "
                )
            if self.matter_power_spectrum is None:
                raise ValueError(
                    "`power_spectrum` cannot be None when `cosmo` is None. "
                )

        degrees = sorted(degrees)

        if self.window is None:
            if f_nl is None:
                b_k = b_1 * np.ones_like(wavenumbers)
            else:
                b_k = b_1 + f_nl * (b_1 - tracer_parameter) \
                    * scale_dependence_modification(
                        self.cosmo, self.redshift
                    )(wavenumbers)

            convolved_multipoles = [
                self.kaiser_factors(ell, b_k, self.growth_rate) \
                * self.matter_power_spectrum(wavenumbers)
                for ell in degrees
            ]

            return convolved_multipoles

        k_interpol = np.logspace(*LOG_K_INTERPOL_RANGE, num=NUM_INTERPOL)

        if f_nl is None:
            b_k = b_1 * np.ones_like(k_interpol)
        else:
            b_k_interpol = b_1 + f_nl * (b_1 - tracer_parameter) \
                * scale_dependence_modification(
                    self.cosmo, self.redshift
                )(k_interpol)

        pk_ell_interpol = {
            ell: self.kaiser_factors(ell, b_k_interpol, self.growth_rate) \
                * self.matter_power_spectrum(k_interpol)
            for ell in degrees
        }

        xi_ell = {
            ell: Spline(
                *P2xi(k_interpol, l=ell, lowring=True)(
                    pk_ell_interpol[ell], extrap=True
                ),
                k=1
            )
            for ell in degrees
        }

        if self.window is not None:
            windows_multipoles = self.window.correlation_multipoles

            s = windows_multipoles['s']

            xi_ell_convolved = {}


                ell: xi_ell[ell](s) \
                    * windows_multipoles['correlation_{}'.format(ell)]
                for ell in degrees
            }
        else:
            xi_ell_convolved = xi_ell


        rr, CF = xi
        return InterpolatedUnivariateSpline(rr, CF)


        return convolved_multipoles

    @staticmethod
    def kaiser_factors(multipole, bias, growth_rate):
        """Return the standard Kaiser power spectrum multipole as a
        multipole of the matter power spectrum.

        Notes
        -----
        `bias` and `growth_rate` must be specified at the same redshift.

        Parameters
        ----------
        multipole : int
            Order of the multipole, ``multipole >= 0``.
        bias : float, array_like
            Scale-independent linear bias of the tracer particles.
        growth_rate : float
            Linear growth rate.

        Returns
        -------
        kaiser_factor : float, array_like
            Standard Kaiser power spectrum multipole factor.

        """
        ell, b, f = multipole, np.atleast_1d(bias), growth_rate

        if ell == 0:
            kaiser_factor = b**2 + 2/3 * b * f + 1/5 * f**2
        elif ell == 2:
            kaiser_factor = 4/3 * b * f + 4/7 * f**2
        elif ell == 4:
            kaiser_factor = 8/35 * f**2
        else:
            kaiser_factor = np.zeros_like(b)

        return np.squeeze(kaiser_factor)
