r"""
Two-point correlator models (:mod:`~harmonia.reader.models`)
===========================================================================

Compute Fourier-space two-point correlator models.


Cartesian multipoles
---------------------------------------------------------------------------

The standard Kaiser model of plane-parallel power spectrum multipoles is
implemented,

.. math::

    \begin{align*}
        P_0(k, z) &=
            \left[
                b_1(z)^2 + \frac{2}{3} b(z, k) f(z) + \frac{1}{5} f(z)^2
            \right] P_\mathrm{m}(k, z) \,, \\
        P_2(k, z) &=
            \left[
                \frac{4}{3} b(z, k) f(z) + \frac{4}{7} f(z)^2
            \right] P_\mathrm{m}(k, z) \,, \\
        P_4(k, z) &= \frac{8}{35} f(z)^2 P_\mathrm{m}(k, z) \,,
    \end{align*}

with shot noise, window convolution and the integral constraint taken
into account [1]_ [2]_; here :math:`f(z)` is the linear growth rate,
:math:`b(z, k) = b_1(z) + f_\textrm{NL} \Delta b(k, z)` is the
scale-dependent linear bias including the modification :math:`\Delta b`
due to local primordial non-Gaussianity :math:`f_\textrm{NL}` (see
:mod:`~harmonia.cosmology.scale_dependence`), and :math:`P_{\textrm{m}}`
is the matter power spectrum.


.. [1] Wilson M. J. et al., 2017. MNRAS 464(3), 3121–3130.
    [arXiv: `1511.07799 <https://arxiv.org/abs/1511.07799>`_]

.. [2] Beutler F. et al., 2017. MNRAS 466(2), 2242–2260.
    [arXiv: `1607.03150 <https://arxiv.org/abs/1607.03150>`_]


Spherical correlator
---------------------------------------------------------------------------

Spherical 2-point correlators are computed from coupling coefficients
(see :mod:`~harmonia.reader.couplings`) as a sum of the signal part

.. math::

    \left\langle \delta_\mu \delta_\nu \right\rangle = \sum_\sigma
        M_{\mu\sigma} M^*_{\nu\sigma} \left[
            b_*(k_\sigma) \Phi_{\mu\sigma} + f_* \Upsilon_{\mu\sigma}
        \right] \left[
            b_*(k_\sigma) \Phi_{\nu\sigma} + f_* \Upsilon_{\nu\sigma}
        \right] \kappa_\sigma^{-1} P_{\textrm{m}*}(k_\sigma) \,,

and the shot noise part

.. math::

    \left\langle \epsilon_\mu \epsilon_\nu \right\rangle =
        \frac{1}{\bar{n}} M_{\mu\nu} \int \operatorname{d}\!r r^2
        (w^2\phi)(r) j_\mu(r) j_\nu(r) \,,

where :math:`*` denotes quantities computed at the fiducial epoch
:math:`z_*`, :math:`\kappa` denotes the normalisation coefficients (see
:mod:`~harmonia.algorithms.discretisation`) and
:math:`j_\mu(r) \equiv j_{\ell_\mu}(k_{\ell_\mu n_\mu} r)`.


.. autosummary::

    CartesianMultipoles
    SphericalCorrelator

|

"""
import logging
import warnings

import numpy as np
from mcfit import P2xi, xi2P
from nbodykit.lab import cosmology
from scipy.integrate import IntegrationWarning
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.algorithms.arrays import CartesianArray, SphericalArray
from harmonia.algorithms.integration import radial_integral
from harmonia.cosmology.geometry import (
    differential_AP_distortion,
    redshift_from_distance,
)
from harmonia.cosmology.scale_dependence import scale_dependence_modification
from harmonia.utils import Progress, mpi_compute, restore_warnings

from .couplings import Couplings, SphericalCoefficientWarning, _group_couplings
from ._kernels import shot_noise_kernel


class CartesianMultipoles:
    """Power spectrum multipoles predicted for a given cosmological model
    and survey specifications.

    Parameters
    ----------
    wavenumbers : float, array_like
        Wavenumbers at which the model is evaluated.
    redshift : float
        Redshift at which the model is evaluated.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology` *or None, optional*
        Baseline cosmological model used to produce the transfer function
        and power spectrum and to compute the linear growth rate.
        This can be subsequently updated when calling
        :meth:`~.convolved_power_multipoles`.  If `None` (default) and not
        subsequently updated, primordial non-Gaussianity modifications
        cannot be computed.
    power_spectrum : callable or None, optional
        Baseline linear matter power spectrum model at `redshift`.
        Ignored when `cosmo` is provided.  This cannot be `None` (default)
        unless it is subsequently updated when calling
        :meth:`~.convolved_power_multipoles`.
    growth_rate : float or None, optional
        Baseline linear growth rate at `redshift`.  If `None` (default),
        this is set by `cosmo` if it is provided; otherwise this is set
        to 0.  This can be subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    mask_multipoles : :class:`numpy.ndarray` *or None, optional*
        Survey mask multipoles given at sampled separations (default
        is `None`).  Orders and sampled separations must be sorted.
    window_multipoles : :class:`~.CartesianArray` *or None, optional*
        Survey window multipoles given at sampled wavenumbers (default
        is `None`).  If provided, integral constraint corrections
        are applied.  Orders and sampled wavenumbers must be sorted.

    Attributes
    ----------
    mask_multipoles : :class:`numpy.ndarray` or None
        Survey mask multipoles given at sampled separations.
    window_multipoles : :class:`~.CartesianArray` or None
        Survey window multipoles given at sampled wavenumbers.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology` or None
        Cosmological model.
    matter_power_spectrum : callable
        Matter power spectrum at `redshift`.
    growth_rate : float
        Linear growth rate at `redshift`.
    attrs : dict
        Any other attributes inherited from input parameters.

    """

    _LOG_K_RANGE_SAMP = -5, 1
    _NUM_SAMP = pow(2, 14)
    _PREDICTED_ORDERS = [0, 2, 4]

    def __init__(self, wavenumbers, redshift, cosmo=None,
                 power_spectrum=None, growth_rate=None,
                 mask_multipoles=None, window_multipoles=None):

        self.attrs = {
            'wavenumbers': wavenumbers,
            'redshift': redshift,
        }

        self._k = wavenumbers
        self._z = redshift
        self._mask_multipoles = mask_multipoles
        self._window_multipoles = window_multipoles

        # Dense sampling wavenumbers for window convolution.
        self._k_samp = np.logspace(*self._LOG_K_RANGE_SAMP, num=self._NUM_SAMP)

        # Set a baseline model.  May be subsequently updated.  Also
        # pre-compute quantities at set wavenumbers.
        self._set_baseline_model(cosmo, power_spectrum, growth_rate)

    def convolved_power_multipoles(self, orders, b_1=None, f_nl=None,
                                   nbar=None, contrast=None, tracer_p=1.,
                                   update_model_kwargs=None):
        """Compute the convolved power spectrum multipoles.

        Parameters
        ----------
        orders : list of int
            Orders of the power spectrum multipoles.  Values only allowed
            from the set {0, 2, 4}.
        b_1 : float or None, optional
            Scale-independent linear bias at input redshift.  If `None`
            (default), no tracer bias is assumed relative to the matter
            power spectrum.
        f_nl : float or None, optional
            Local primordial non-Gaussianity (default is `None`).
        nbar : float or None, optional
            Mean particle number density (in cubic :math:`h`/Mpc).  If
            `None` (default), shot noise is neglected.
        contrast : float or None, optional
            If not `None` (default), this adds additional shot noise
            ``1 / (contrast * nbar)`` from a FKP-style random catalogue.
        tracer_p : float, optional
            Tracer-dependent parameter for bias modulation by `f_nl`
            (default is 1.).
        update_model_kwargs : dict or None, optional
            Parameters `cosmo`, `power_spectrum` and `growth_rate` passed
            as keyword arguments to update the baseline cosmological model.

        Returns
        -------
        convolved_power : :class:`~.algorithms.arrays.CartesianArray`
            Convolved power spectrum multipoles.

        """
        # Updated initial baseline cosmological model if relevant
        # arguments specified.
        if update_model_kwargs is not None:
            self._set_baseline_model(**update_model_kwargs)

        b_1 = b_1 or 1.

        if f_nl is not None and self.cosmo is None:
            raise TypeError(
                "Cannot accept `f_nl` values without input `cosmo`."
            )

        alpha = 0. if contrast is None else 1 / contrast
        shot_noise = (1 + alpha) / nbar

        convolved_power = CartesianArray(
            orders, self._k, shot_noise=shot_noise
        )

        # If no survey mask, no convolution is needed.
        if self._mask_multipoles is None:
            b_k = b_1 * np.ones_like(self._k)
            if f_nl:
                b_k += self._mode_modification * f_nl * (b_1 - tracer_p)

            convolved_power[:] = np.concatenate([
                self.kaiser_factors(ell, b_k, self.growth_rate)
                * self._mode_powers
                + np.equal(ell, 0) * shot_noise
                for ell in orders
            ])

            return convolved_power

        # Get mask multipoles as a directory.
        s_mask = np.unique(self._mask_multipoles['separation'])
        mask = {
            ell: self._mask_multipoles['correlation'][
                self._mask_multipoles['order'] == ell
            ]
            for ell in np.unique(self._mask_multipoles['order'])
        }

        # Get power multipoles at densely sampled points for interpolation.
        b_k_samp = b_1 * np.ones_like(self._k_samp)
        if f_nl:
            b_k_samp += self._mode_modification_samp * f_nl * (b_1 - tracer_p)

        power_samp = {
            ell: self.kaiser_factors(ell, b_k_samp, self.growth_rate) \
                * self.matter_power_spectrum(self._k_samp)
            for ell in self._PREDICTED_ORDERS
        }

        # Hankel transform to correlation multipoles.
        with warnings.catch_warnings():  # suppress `mcfit` warning
            warnings.filterwarnings(
                'ignore',
                message=(
                    "The default value of extrap has been changed to False, "
                    "set it to True if you cannot reproduce previous results"
                )
            )
            correlation = {
                ell: Spline(
                    *P2xi(self._k_samp, l=ell, lowring=True)(
                        power_samp[ell], extrap=False
                    ), k=1
                )(s_mask)
                for ell in self._PREDICTED_ORDERS
            }

        # Convolve correlation multipoles.
        correlation_convolved = {
            ell: self._convolve(ell, correlation, mask) for ell in orders
        }

        # Hankel transform back to power multipoles.
        with warnings.catch_warnings():  # suppress warnings from `mcfit`
            warnings.filterwarnings(
                'ignore',
                message=(
                    "The default value of extrap has been changed to False, "
                    "set it to True if you cannot reproduce previous results"
                )
            )
            power_conv = {
                ell: Spline(
                    *xi2P(s_mask, l=ell, lowring=True)(
                        correlation_convolved[ell], extrap=False
                    ), k=1
                )(self._k) + np.equal(ell, 0) * shot_noise
                for ell in orders
            }

        # Apply integral constraint (ic) corrections.
        if self._window_multipoles is None:
            warnings.warn(
                "`window_multipoles` attribute is missing. "
                "Integral constraint is not corrected."
            )
        else:
            # Get window multipoles.
            k_win = np.unique(
                self._window_multipoles.array['wavenumber']
            )
            window = {
                ell: self._window_multipoles.array['power'][
                    self._window_multipoles.array['order'] == ell
                ]
                for ell in np.unique(self._window_multipoles.array['order'])
            }

            # Interpolate window multipoles at model wavenumbers.
            window_intrpl = {
                ell: Spline(k_win, window[ell])(self._k)
                for ell in power_conv
            }

            with warnings.catch_warnings():  # suppress warnings from `mcfit`
                warnings.filterwarnings(
                    'ignore',
                    message=(
                        "The default value of extrap has been changed "
                        "to False, set it to True if you cannot "
                        "reproduce previous results"
                    )
                )
                ic_correction = Spline(
                    *xi2P(s_mask, l=0, lowring=True)(
                        correlation_convolved[0], extrap=False
                    ), k=1
                )(0)

            power_conv.update({
                ell: power_conv[ell] - ic_correction * window_intrpl[ell]
                for ell in orders
            })

        convolved_power[:] = np.concatenate(
            [power_conv[ell] for ell in orders]
        )

        return convolved_power

    @staticmethod
    def kaiser_factors(ell, b, f):
        """Return the standard Kaiser power spectrum multipole as a
        multipole of the matter power spectrum.

        Notes
        -----
        Linear bias `b` and growth rate `f` must be specified at the
        same redshift.

        Parameters
        ----------
        ell : int
            Order of the multipole in {0, 2, 4}.
        b : float, array_like
            Linear bias of the tracer particles.
        f : float or None
            Linear growth rate.

        Returns
        -------
        float, array_like
            Standard Kaiser factor to multiply by the matter power
            spectrum.

        """
        if ell == 0:
            factor = b**2 + 2./3. * b * f + 1./5. * f**2
        elif ell == 2:
            factor = 4./3. * b * f + 4./7. * f**2
        elif ell == 4:
            factor = 8./35. * f**2
        else:
            raise ValueError("Multipole order must be one of {0, 2, 4}.")

        return factor

    def _set_baseline_model(self, cosmo=None, power_spectrum=None,
                            growth_rate=None):

        # If `cosmo` is not None, `self._kernel` as A(k) (see docstring),
        # `self._mode_modification` is its evaluation at wavenumbers.
        # `self._mode_powers` is the evaluated matter power spectrum.
        self.cosmo = cosmo
        if self.cosmo is None:
            self.matter_power_spectrum = power_spectrum
            self.growth_rate = growth_rate or 0.
            self._kernel = None
            self._mode_modification = None
            self._mode_modification_samp = None
        else:
            self.matter_power_spectrum = cosmology.LinearPower(
                self.cosmo, redshift=self._z, transfer='CLASS'
            )
            self.growth_rate = growth_rate if growth_rate is not None \
                else self.cosmo.scale_independent_growth_rate(self._z)
            self._kernel = scale_dependence_modification(self.cosmo, self._z)
            self._mode_modification = self._kernel(self._k)
            self._mode_modification_samp = self._kernel(self._k_samp)

        # (Re-)set pre-computed quantities.
        try:
            self._mode_powers = self.matter_power_spectrum(self._k)
            self._mode_powers_samp = self.matter_power_spectrum(self._k_samp)
        except TypeError:
            warnings.warn(
                "Cannot evaluate :attr:`matter_power_spectrum`. "
                "No baseline model is current set."
            )

    @staticmethod
    def _convolve(order, correlation, mask):

        if order == 0:
            return correlation[0] * mask[0] \
                + 1./5. * correlation[2] * mask[2] \
                + 1./9. * correlation[4] * mask[4]

        if order == 2:
            return correlation[0] * mask[2] \
                + correlation[2] * (
                    mask[0] + 2./7. * mask[2] + 2./7. * mask[4]
                ) \
                + correlation[4] * (
                    2./7. * mask[2] + 100./693. * mask[4] + 25./143. * mask[6]
                )

        if order == 4:
            return correlation[0] * mask[4] \
                + correlation[2] * (
                    18./35. * mask[2] + 20./77. * mask[4] + 45./143. * mask[6]
                ) \
                + correlation[4] * (
                    mask[0]
                    + 20./77. * mask[2] + 162./1001. * mask[4]
                    + 20./143. * mask[6] + 490./2431. * mask[8]
                )

        raise ValueError(f"Order not predicted by the model: {order}.")


class SphericalCorrelator:
    """Spherical two-point correlator predicted for a given cosmological
    model and survey specifications.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    redshift : float
        Redshift at which the model is evaluated.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology` *or None, optional*
        Baseline cosmological model used to produce the transfer function
        (and the power spectrum and linear growth rate if these are not
        set but required in model evaluation).  This can be subsequently
        updated when calling :meth:`~.two_point_correlator`
        or :meth:`~.correlator_matrix`.  If `None` (default) and not
        subsequently updated, primordial non-Gaussianity modifications
        cannot be computed.
    power_spectrum : callable or None, optional
        Baseline linear matter power spectrum model at `redshift`.
        Ignored when `cosmo` is provided; otherwise this cannot be `None`
        (default) unless it is subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    growth_rate : float or None, optional
        Baseline linear growth rate at `redshift`.  If `None` (default),
        this is set by `cosmo` (if provided); otherwise this is set
        to zero.  This can be subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    couplings : :class:`~.couplings.Couplings` *or None, optional*
        Baseline coupling coefficients consistent with the underlying
        cosmological model `cosmo`.  If `None` (default), this is compiled
        from `survey_specs` and `cosmo_specs` if either is provided;
        otherwise all couplings are assumed to be trivial (i.e. angular
        and radial couplings are Kronecker deltas).  This can be
        subsequently updated when calling :meth:`~.two_point_correlator`
        or :meth:`~.correlator_matrix`.
    survey_specs : dict{str: callable or None} or None, optional
        Survey specification functions to be passed as `survey_specs` to
        :class:`~harmonia.reader.couplings.Couplings` when couplings are
        compiled.  Also used in shot noise calculations.
    cosmo_specs : dict{str: callable, bool or None} or None, optional
        Baseline cosmological specification functions to be passed as
        `cosmo_specs` to :class:`~harmonia.reader.couplings.Couplings`
        when couplings are compiled.  If not `None` (default), it must be
        a dictionary holding keys listed in :class:`~.couplings.Couplings`:
        if callable values are passed to the keys ``'chi_of_z'``,
        ``'clustering_evolution'``, ``'growth_evolution'`` or
        ``'differential_AP_distortion'``, they should be consistent with
        the current :attr:`cosmo`; otherwise `True` can be passed here and
        their values are then derived from `cosmo` (which must then be
        set), or unspecified keys assume `None` values; also note some
        keys are linked and values must be simultaneously provided.  This
        can be subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

    Attributes
    ----------
    couplings : :class:`nbodykit.cosmology.cosmology.Cosmology` or None
        Spherical coupling coefficients.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology` or None
        Cosmological model.
    matter_power_spectrum : callable
        Matter power spectrum at `redshift`.
    growth_rate : float
        Linear growth rate at `redshift`.
    attrs : dict
        Any other attributes inherited from input parameters.

    See Also
    --------
    :class:`~harmonia.reader.couplings.Couplings`
        More details related to `couplings` and especially the
        `cosmo_specs` parameter.

    """

    def __init__(self, disc, redshift, cosmo=None, power_spectrum=None,
                 growth_rate=None, couplings=None,
                 survey_specs=None, cosmo_specs=None, comm=None):

        self.comm = comm
        self.logger = logging.getLogger(self.__class__.__name__)

        self.attrs = {
            'disc': disc,
            'redshift': redshift,
        }

        self._disc = disc
        self._z = redshift

        # NOTE: This assignment is needed to help :meth:`_render_cosmo_specs`
        # set the couplings next, which are not initialised with `cosmo`,
        # whose formal assignment must come after in
        # :meth:`_set_baseline_model`, as therein the couplings are processed.
        self.cosmo = cosmo

        self._survey_specs = survey_specs
        self._cosmo_specs = self._render_cosmo_specs(cosmo_specs)

        if couplings is None:
            self.couplings = Couplings(
                self._disc,
                survey_specs=self._survey_specs,
                cosmo_specs=self._cosmo_specs,
                initialise=True
            )
        else:
            self.couplings = couplings

        # Group couplings for mass access.
        self._grouped_couplings = _group_couplings(self.couplings)

        # Set a baseline model and pre-compute power and transfer function for
        # fixed modes.  May be subsequently updated.
        # NOTE: This step must come after setting the `couplings` attribute,
        # since it also (re-)processes the radial/RSD couplings.
        self._set_baseline_model(
            cosmo=cosmo, power_spectrum=power_spectrum, growth_rate=growth_rate
        )

        # Compile shot noise levels and cosmology-independent angular
        # coupling sums as directories to reduce computational redundancy.
        self._shot_noise_levels = self._compile_shot_noise_levels()
        if self.couplings is None:
            self._angular_sums = None
        else:
            self._angular_sums = self._compile_angular_sums()

    def two_point_correlator(self, mu, nu, b_1, f_nl=None, nbar=None,
                             contrast=None, tracer_p=1.,
                             update_model_kwargs=None):
        """Compute two-point correlator for given triplet indices.

        Parameters
        ----------
        mu, nu : tuple(int, int, int)
            Coefficient triplet index.
        b_1 : float
            Scale-independent linear bias at input redshift.
        f_nl : float or None, optional
            Local primordial non-Gaussianity (default is `None`).
        nbar : float or None, optional
            Mean particle number density (in cubic :math:`h`/Mpc).  If
            `None` (default), shot noise is neglected.
        contrast : float or None, optional
            If not `None` (default), this adds additional shot noise
            ``1 / (contrast * nbar)`` from a FKP-style random catalogue.
        tracer_p : float, optional
            Tracer-dependent parameter for bias modulation by `f_nl`
            (default is 1.).
        update_model_kwargs : dict or None, optional
            Parameters `cosmo`, `power_spectrum`, `growth_rate` and
            `cosmo_specs` passed as keyword arguments to update the
            baseline cosmological model.  If `cosmo_specs` is passed
            (including `None` values), radial and RSD couplings will be
            updated.

        Returns
        -------
        complex
            Spherical Fourier 2-point function value for given triplet
            indices.  If `cosmo_specs` is passed, radial and RSD couplings
            will be updated.

        """
        if update_model_kwargs is not None:
            if 'cosmo_specs' in update_model_kwargs:
                update_model_kwargs.update({'update_couplings': True})
            self._set_baseline_model(**update_model_kwargs)

        f = self.growth_rate
        # Grand sum over dummy degree index, say `ell_sigma`.
        correlator_value = 0.
        for ell_idx, ell_sigma in enumerate(self._disc.degrees):
            # Fetch angular sum over order `m_sigma`.
            try:
                angular_sum = self._get_angular_sums(mu, nu)[ell_sigma]
            except TypeError:
                # Non-subscriptable 0.'s and 1.'s independent of degree.
                angular_sum = self._get_angular_sums(mu, nu)

            # Perform radial sum over depth `n_sigma`.
            b_k = b_1
            if f_nl:
                b_k += f_nl  * (b_1 - tracer_p) \
                    * self._mode_modification[ell_idx]
            if f:
                radial_sum = np.sum(
                    (
                        b_k
                        * self._get_grouped_couplings('radial', mu)[ell_idx]
                        + f
                        * self._get_grouped_couplings('rsd', mu)[ell_idx]
                    ) * (
                        b_k
                        * self._get_grouped_couplings('radial', nu)[ell_idx]
                        + f
                        * self._get_grouped_couplings('rsd', nu)[ell_idx]
                    ) * self._normalised_mode_powers[ell_idx]
                )
            else:
                radial_sum = np.sum(
                    b_k * self._get_grouped_couplings('radial', mu)[ell_idx]
                    * b_k * self._get_grouped_couplings('radial', nu)[ell_idx]
                    * self._normalised_mode_powers[ell_idx]
                )

            correlator_value += angular_sum * radial_sum

        if nbar is not None:
            alpha = 0. if contrast is None else 1 / contrast
            correlator_value += (1 + alpha) / nbar \
                * self._get_shot_noise_level(mu, nu)

        return correlator_value

    def correlator_matrix(self, pivot, b_1=None, f_nl=None, nbar=None,
                          contrast=None, tracer_p=1., diagonal=False,
                          shot_noise_only=False, update_model_kwargs=None,
                          report_progress=False):
        """Compute two-point correlator matrix for some vetorisation of all
        spectrum modes.

        Parameters
        ----------
        pivot : {'natural', 'spectral'}
            Pivot order for vectorisation.
        b_1 : float
            Scale-independent linear bias at input redshift.
        f_nl : float or None, optional
            Local primordial non-Gaussianity (default is `None`).
        nbar : float or None, optional
            Mean particle number density (in cubic :math:`h`/Mpc).  If
            `None` (default), shot noise is neglected.
        contrast : float or None, optional
            If not `None` (default), this adds additional shot noise
            ``1 / (contrast * nbar)`` from a FKP-style random catalogue.
        tracer_p : float, optional
            Tracer-dependent parameter for bias modulation by `f_nl`
            (default is 1.).
        diagonal : bool, optional
            If `True` (default is `False`), return only the diagonal
            matrix part.
        shot_noise_only : bool, optional
            If `True` (default is `False`), return only the shot noise
            correlator matrix.
        update_model_kwargs : dict or None, optional
            Parameters `cosmo`, `power_spectrum`, `growth_rate` and
            `cosmo_specs` passed as keyword arguments to update the
            baseline cosmological model.  If `cosmo_specs` is passed
            (including `None` values), radial and RSD couplings will be
            updated.
        report_progress : bool, optional
            If `True` (default is `False`), progress status will be
            reported.

        Returns
        -------
        complex :class:`numpy.ndarray`
            Two-point correlator matrix vectorised by the given pivot.

        See Also
        --------
        :class:`~harmonia.algorithms.arrays.SphericalArray`

        """
        if update_model_kwargs is not None:
            if 'cosmo_specs' in update_model_kwargs:
                update_model_kwargs.update({'update_couplings': True})
            self._set_baseline_model(**update_model_kwargs)

        index_array = SphericalArray(self._disc)
        # NOTE: Careful sorting is needed here.
        if pivot == 'natural':
            index_vector = [
                tuple(index)
                for index in index_array.array['index']
            ]
        elif pivot == 'spectral':
            index_vector = [
                tuple(index)
                for index in index_array.array['index'][
                    np.argsort(index_array.array['wavenumber'])
                ]
            ]
        else:
            raise ValueError(f"Invalid pivot option: {pivot}.")

        dim = len(index_vector)

        ## Diagonal: Only return diagonal matrix.

        if diagonal:
            # Shot noise contribution.
            if nbar is None:
                shot_noise_diag = np.zeros(dim, dtype=complex)
            else:
                alpha = 0. if contrast is None else 1 / contrast
                shot_noise_diag = (1 + alpha) / nbar * np.asarray([
                    self._get_shot_noise_level(mu, mu)
                    for mu in index_vector
                ])

            if shot_noise_only:
                return np.diag(shot_noise_diag)

            # Signal contribution.
            signal_diag = np.asarray([
                self.two_point_correlator(
                    mu, mu, b_1=b_1, f_nl=f_nl, tracer_p=tracer_p
                )
                for mu in index_vector
            ])

            two_point_corr_diag = signal_diag + shot_noise_diag

            return np.diag(two_point_corr_diag)

        ## Full: Return the entire correlator matrix.

        # Shot noise contribution.
        shot_noise_mat = np.zeros((dim,) * 2, dtype=complex)

        if nbar is not None:
            alpha = 0. if contrast is None else 1 / contrast
            shot_noise_amplitude = (1 + alpha) / nbar
            for row_idx in range(dim):
                for col_idx in range(row_idx + 1):
                    mu, nu = index_vector[row_idx], index_vector[col_idx]
                    shot_noise_mat[row_idx, col_idx] = shot_noise_amplitude * \
                        self._get_shot_noise_level(mu, nu)

        if shot_noise_only:
            return shot_noise_mat

        # Signal contribution.
        if report_progress:
            progress = Progress(
                (1 + dim) * dim / 2,
                process_name="correlator matrix",
                logger=self.logger, comm=self.comm
            )
            idx = 0
        else:
            progress = None

        signal_mat = np.zeros((dim,) * 2, dtype=complex)
        for row_idx in range(dim):
            for col_idx in range(row_idx + 1):
                mu, nu = index_vector[row_idx], index_vector[col_idx]
                signal_mat[row_idx, col_idx] = self.two_point_correlator(
                    mu, nu, b_1=b_1, f_nl=f_nl, tracer_p=tracer_p
                )
                # pylint: disable=multiple-statements
                if progress:
                    progress.report(idx); idx += 1

        # Use Hermitian property to fill in the strictly upper triangular part.
        two_point_corr_mat = signal_mat + shot_noise_mat
        two_point_corr_mat[np.triu_indices(dim, k=1)] = np.conj(
            np.transpose(two_point_corr_mat)
        )[np.triu_indices(dim, k=1)]

        return two_point_corr_mat

    def radialised_power(self, b_1=None, f_nl=None, nbar=None, contrast=None,
                         tracer_p=1., shot_noise_only=False):
        """Compute the radialised spherical mode power.

        Notes
        -----
        This relies on :meth:`~.correlator_matrix` with ``diagonal=True``.
        Results are only meaningful in the radialisation limit when
        all coupling coefficients are trivial (i.e. Kronecker deltas) so
        that spherical modes are mutually independent [3]_.  Mode power is
        averaged over equivalent spherical orders and suitably normalised
        so that it matches the Cartesian power spectrum for a full-sky
        statistically isotripoc map.

        Parameters
        ----------
        b_1 : float
            Scale-independent linear bias at input redshift.
        f_nl : float or None, optional
            Local primordial non-Gaussianity (default is `None`).
        nbar : float or None, optional
            Mean particle number density (in cubic :math:`h`/Mpc).  If
            `None` (default), shot noise is neglected.
        contrast : float or None, optional
            If not `None` (default), this adds additional shot noise
            ``1 / (contrast * nbar)`` from a FKP-style random catalogue.
        tracer_p : float, optional
            Tracer-dependent parameter for bias modulation by `f_nl`
            (default is 1.).
        shot_noise_only : bool, optional
            If `True` (default is `False`), return only the shot noise
            correlator matrix.

        Returns
        -------
        dict
            Radialised spherical mode powers at wavenumbers with mode
            indices.

        See Also
        --------
        :attr:`~harmonia.mapper.map_transform.SphericalMap.mode_power`


        .. [3] Rassat A. & Refregier A., 2012. A&A 540, A115.
           [arXiv: `1112.3100 <https://arxiv.org/abs/1112.3100>`_]

        """
        diagonal_correlator = self.correlator_matrix(
            'natural', b_1=b_1, f_nl=f_nl, nbar=nbar, contrast=contrast,
            tracer_p=tracer_p, diagonal=True, shot_noise_only=shot_noise_only
        )

        index_vector = self._gen_operable_indices()

        normalisation_vector = [
            self._disc.normalisations[index[0], index[-1]]
            for index in index_vector
        ]

        mode_power_array = SphericalArray(self._disc)

        mode_power_array[:] = np.multiply(
            normalisation_vector, np.diag(diagonal_correlator)
        )

        mode_powers = mode_power_array.vectorise('spectral', collapse='mean')
        wavenumbers = np.sort(list(self._disc.wavenumbers.values()))

        return {
            'wavenumbers': wavenumbers,
            'mode_powers': np.real(mode_powers),
        }

    def _set_baseline_model(self, **kwargs):
        """(Re-)set baseline model.

        Parameters
        ----------
        **kwargs
            Parameters `cosmo`, `power_spectrum`, `growth_rate`,
            `cosmo_specs` and `update_couplings` to be set or reset.

        """
        ## `self._kernel` is A(k) (see docstring), `self._mode_modification`
        ## is its evaluation at wavenumbers, and `self._normalised_mode_powers`
        ## is the evaluated matter power spectrum normalised 'spherically'.

        if 'cosmo' in kwargs:
            self.cosmo = kwargs['cosmo']
            # (Re-)set pre-computed quantities.
            if self.cosmo is None:
                self._kernel = None
                self._mode_modification = None
            else:
                self._kernel = scale_dependence_modification(
                    self.cosmo, self._z
                )
                self._mode_modification = [
                    np.asarray([
                        self._kernel(
                            self._disc.wavenumbers[ell_sigma, n_sigma]
                        )
                        for n_sigma in range(1, nmax_sigma + 1)
                    ])
                    for ell_sigma, nmax_sigma in zip(
                        self._disc.degrees, self._disc.depths
                    )
                ]

        if 'power_spectrum' in kwargs:
            if self.cosmo is None:
                self.matter_power_spectrum = kwargs['power_spectrum']
            else:
                self.matter_power_spectrum = cosmology.LinearPower(
                    self.cosmo, redshift=self._z, transfer='CLASS'
                )
            # (Re-)set pre-computed quantities.
            try:
                self._normalised_mode_powers = [
                    np.asarray([
                        self.matter_power_spectrum(
                            self._disc.wavenumbers[ell_sigma, n_sigma]
                        ) / self._disc.normalisations[ell_sigma, n_sigma]
                        for n_sigma in range(1, nmax_sigma + 1)
                    ])
                    for ell_sigma, nmax_sigma in zip(
                        self._disc.degrees, self._disc.depths
                    )
                ]
            except TypeError:
                warnings.warn(
                    "Cannot evaluate :attr:`matter_power_spectrum`. "
                    "No baseline model is current set."
                )

        if 'growth_rate' in kwargs:
            if self.cosmo is None:
                self.growth_rate = kwargs['growth_rate'] or 0.
            else:
                self.growth_rate = kwargs['growth_rate'] \
                    if kwargs['growth_rate'] is not None \
                    else self.cosmo.scale_independent_growth_rate(self._z)

        ## Only necessary to update radial and RSD couplings (based on the
        ## latest `cosmo` attribute) when `cosmo_specs` is specified.  Any
        ## previously computed angular couplings are attached.

        if kwargs.get('update_couplings'):
            try:
                self._cosmo_specs = \
                    self._render_cosmo_specs(kwargs['cosmo_specs'])
            except KeyError:
                warnings.warn(
                    "`update_couplings` is ignored as `cosmo_specs` was not "
                    "passed. If you intend it to be `None` it still needs to "
                    "be explicitly passed."
                )
            else:
                try:
                    current_angular_couplings = \
                        self.couplings.couplings['angular']
                except (AttributeError, KeyError):
                    current_angular_couplings = None
                self.couplings = Couplings(
                    self._disc,
                    survey_specs=self._survey_specs,
                    cosmo_specs=self._cosmo_specs,
                    external_angular_couplings=current_angular_couplings,
                    initialise=True
                )
                self._grouped_couplings = _group_couplings(self.couplings)

    def _render_cosmo_specs(self, cosmo_specs):

        # This renders `cosmo_specs` ready to be passed to
        # :class:`~.couplings.Couplings` for coupling compilation.

        _error_msg = (
            "Cannot convert '{}' from True to callable as "
            "the `cosmo` attribute is not set."
        )

        if cosmo_specs is None:
            return None

        if isinstance(cosmo_specs, dict):
            # pylint: disable=protected-access
            _cosmo_specs = dict.fromkeys(Couplings._cosmo_specs.keys())
            if not set(cosmo_specs.keys()).issubset(_cosmo_specs.keys()):
                raise ValueError(
                    "Invalid `cosmo_specs` keys: {}.".format(
                        set(cosmo_specs.keys()) - set(_cosmo_specs.keys())
                    )
                )
            _cosmo_specs.update(cosmo_specs)

            # Convert `True` values using the `cosmo` attribute.
            if _cosmo_specs['z_from_r'] is True:
                try:
                    z_from_r = redshift_from_distance(self.cosmo)
                except AttributeError:
                    raise ValueError(_error_msg.format('z_from_r'))
                else:
                    _cosmo_specs.update({'z_from_r': z_from_r})

            if _cosmo_specs['clustering_evolution'] is True:
                try:
                    clustering_evolution = lambda z: \
                        self.cosmo.scale_independent_growth_factor(z) \
                        / self.cosmo.scale_independent_growth_factor(self._z)
                except AttributeError:
                    raise ValueError(_error_msg.format('clustering_evolution'))
                else:
                    _cosmo_specs.update({
                        'clustering_evolution': clustering_evolution
                    })

            if _cosmo_specs['growth_evolution'] is True:
                try:
                    growth_evolution = lambda z: \
                        self.cosmo.scale_independent_growth_rate(z) \
                        / self.cosmo.scale_independent_growth_rate(self._z)
                except AttributeError:
                    raise ValueError(_error_msg.format('growth_evolution'))
                else:
                    _cosmo_specs.update({'growth_evolution': growth_evolution})

            if _cosmo_specs['differential_AP_distortion'] is True:
                try:
                    diff_AP = differential_AP_distortion(
                        _cosmo_specs['chi_of_z'], self.cosmo.comoving_distance
                    )
                except AttributeError:
                    raise ValueError(
                        _error_msg.format('differential_AP_distortion')
                    )
                else:
                    _cosmo_specs.update({
                        'differential_AP_distortion': diff_AP
                    })

            return _cosmo_specs

        raise TypeError("Invalid `cosmo_specs` passed.")

    def _get_grouped_couplings(self, coupling_type, mu):

        return self._grouped_couplings[coupling_type][mu[0], mu[-1]]

    def _get_angular_sums(self, mu, nu):

        try:
            return self._angular_sums[(mu[0], mu[1]), (nu[0], nu[1])]
        except (KeyError, TypeError):
            try:  # use Hermitian conjugate if unavailable
                return self._angular_sums[(nu[0], nu[1]), (mu[0], mu[1])]
            except TypeError:  # non-existent, reduce to Kronecker delta
                return float(mu[0] == nu[0] and mu[1] == nu[1])

    def _get_shot_noise_level(self, mu, nu):

        try:
            return self._shot_noise_levels[mu, nu]
        except KeyError:  # use Hermitian conjugate if unavailable
            return self._shot_noise_levels[nu, mu]

    def _compile_angular_sums(self):

        # Compiles all index pairs of the form (\ell_\mu, m_\mu, \ell_\nu,
        # m_\nu) where \nu is a higher index than \mu (the other cases are
        # obtained by Hermitian conjuate).
        index_pair_vector = list(
            (first_index, second_index)
            for first_index in self._gen_operable_indices(subtype='angular')
            for second_index in self._gen_operable_indices(
                subtype='angular', above_from=first_index
            )
        )

        if self.comm is None or self.comm.rank == 0:
            self.logger.info("Compiling fixed angular sums...")

        # Compile, for each index pair of the form above, all its angular sums
        # indexed by a new degree index (say, `ell_sigma`) of the form:
        # \sum_{m_\ell} M_{\ell_\mu, m_\mu, \ell_\sigma, m_\sigma)
        # * M_{\ell_\nu, m_\nu, \ell_\sigma, m_\sigma)^*.
        angular_sums_by_index_pair = mpi_compute(
            index_pair_vector, self._compile_angular_sums_by_index_pair,
            comm=self.comm, process_name="angular sum compilation"
        )

        # Store compiled results in a directory, where each key is an index
        # pair of the form above, and each entry is a dictionary of angular
        # sums indexed by the new degree index (i.e. `ell_sigma`).
        angular_sums = {}
        for index_pair, angular_sums_for_index_pair \
                in zip(index_pair_vector, angular_sums_by_index_pair):
            angular_sums.update({
                tuple(index_pair): angular_sums_for_index_pair
            })

        if self.comm is None or self.comm.rank == 0:
            self.logger.info("... compiled fixed angular sums.")

        return angular_sums

    def _compile_shot_noise_levels(self):

        # Compiles all index pairs (`mu`, `nu`) where `nu` is a higher index
        # than `mu` (the other cases are obtained by Hermitian conjuate).
        index_pair_vector = list(
            (first_index, second_index)
            for first_index in self._gen_operable_indices()
            for second_index in \
                self._gen_operable_indices(above_from=first_index)
        )

        if self.comm is None or self.comm.rank == 0:
            self.logger.info("Compiling shot noise levels...")

        # Compile, for each index pair of the form above, the shot noise level
        # of the form: M_{\mu, \nu} * shot noise integral[selection, weight].
        shot_noise_levels_by_index_pair = mpi_compute(
            index_pair_vector, self._compile_shot_noise_levels_by_index_pair,
            comm=self.comm, process_name="shot noise level compilation"
        )

        # Store compiled results in a directory, where each key is an index
        # pair of the form above, and each entry is a complex shot noise
        # integral value including the mask.
        shot_noise_levels = {}
        with warnings.catch_warnings(record=True) as any_warnings:
            for index_pair, shot_noise_levels_for_index_pair \
                    in zip(index_pair_vector, shot_noise_levels_by_index_pair):
                shot_noise_levels.update({
                    tuple(index_pair): shot_noise_levels_for_index_pair
                })
        if any_warnings:
            restore_warnings(any_warnings)
            warnings.warn(
                "Poor shot noise integration convergence.",
                IntegrationWarning
            )

        if self.comm is None or self.comm.rank == 0:
            self.logger.info("... compiled shot noise levels.")

        return shot_noise_levels

    def _compile_angular_sums_by_index_pair(self, index_pair):

        mu, nu = index_pair

        # For each degree index (say `ell_sigma`), return \sum_{m_\sigma}
        # M_{\mu, \sigma} * M_{\nu, \sigma}^*.
        angular_sums_for_index_pair_by_degree = {
            ell_sigma: sum([
                self.couplings['angular', mu, (ell_sigma, m_sigma)]
                * np.conj(self.couplings['angular', nu, (ell_sigma, m_sigma)])
                for m_sigma in range(- ell_sigma, ell_sigma + 1)
            ])
            for ell_sigma in self._disc.degrees
        }

        return angular_sums_for_index_pair_by_degree

    def _compile_shot_noise_levels_by_index_pair(self, index_pair):

        mu, nu = index_pair

        M_mu_nu = self.couplings['angular', mu, nu]

        k_mu = self._disc.wavenumbers[mu[0], mu[-1]]
        k_nu = self._disc.wavenumbers[nu[0], nu[-1]]

        # Return the radial integral over selection(r) * weight(r)^2 *
        # j_{\ell_mu}(k_\mu r) * j_{\ell_\nu}(k_\nu r).
        with warnings.catch_warnings(record=True) as any_warning:
            shot_noise_integral = radial_integral(
                lambda r: shot_noise_kernel(
                    r, mu, nu, k_mu, k_nu,
                    selection=getattr(self._survey_specs, 'selection', None),
                    weight=getattr(self._survey_specs, 'weight', None)
                ), self._disc.attrs['boundary_radius']
            )
        if any_warning and not np.isclose(shot_noise_integral, 0.):
            warnings.warn(
                "Poorly determined shot noise coefficient "
                "for index pair {} and {}.".format(mu, nu),
                category=SphericalCoefficientWarning
            )

        return M_mu_nu * shot_noise_integral

    def _gen_operable_indices(self, subtype=None, above_from=None):

        if subtype == 'angular':
            operable_indices = [
                (ell, m)
                for ell in self._disc.degrees
                for m in range(- ell, ell + 1)
            ]
        elif subtype == 'radial':
            operable_indices = [
                (ell, n)
                for ell, nmax in zip(self._disc.degrees, self._disc.depths)
                for n in range(1, nmax + 1)
            ]
        else:
            operable_indices = [
                (ell, m, n)
                for ell, nmax in zip(self._disc.degrees, self._disc.depths)
                for m in range(- ell, ell + 1)
                for n in range(1, nmax + 1)
            ]

        if above_from is not None:
            operable_indices = [
                index for index in operable_indices if index >= above_from
            ]

        return operable_indices
