r"""
Two-point correlator models (:mod:`~harmonia.reader.models`)
===========================================================================

Compute Fourier-space two-point correlator models.

.. autosummary::

    CartesianMultipoles
    SphericalCorrelator

|

"""
import logging
import warnings
from itertools import product

import numpy as np
from mcfit import P2xi, xi2P
from nbodykit.lab import cosmology
from scipy.integrate import IntegrationWarning
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.algorithms.arrays import CartesianArray, SphericalArray
from harmonia.algorithms.integration import radial_integral
from harmonia.cosmology.scale_dependence import scale_dependence_modification
from harmonia.utils import mpi_compute, restore_warnings

from .couplings import Couplings
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
        Cosmological model used to produce the transfer function and
        power spectrum and to compute the linear growth rate.  If `None`
        (default) and not subsequently provided when calling some of the
        methods, non-Gaussianity modifications cannot be computed.  This
        can be subsequently updated when calling
        :meth:`~.convolved_power_multipoles`.
    power_spectrum : callable or None, optional
        Linear matter power spectrum model at `redshift`.  When `cosmo`
        is provided, this is ignored; otherwise this cannot be `None`
        unless it is subsequently provided when calling some of the
        methods.  This can be subsequently updated when calling
        :meth:`~.convolved_power_multipoles`.
    growth_rate : float or None, optional
        Linear growth rate at `redshift`.  If `None` (default), this is
        set by `cosmo` if it is provided; otherwise this is set to 0.
        This can be subsequently updated when calling
        :meth:`~.convolved_power_multipoles`.
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

    def __init__(self, wavenumbers, redshift, cosmo=None, power_spectrum=None,
                 growth_rate=None,
                 mask_multipoles=None, window_multipoles=None):

        self.attrs = {
            'wavenumbers': wavenumbers,
            'redshift': redshift
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

    def convolved_power_multipoles(self, orders, b_1, f_nl=None, nbar=None,
                                   contrast=None, tracer_p=1.,
                                   baseline_model_kwargs=None):
        """Compute the convolved power spectrum multipoles.

        Parameters
        ----------
        orders : list of int
            Orders of the power spectrum multipoles.  Values only allowed
            from the set {0, 2, 4}.
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
        baseline_model_kwargs : dict or None, optional
            Parameters `cosmo`, `power_spectrum` and `growth_rate` passed
            as keyword arguments to reset the baseline cosmological model.

        Returns
        -------
        convolved_power : :class:`~.algorithms.arrays.CartesianArray`
            Convolved power spectrum multipoles.

        """
        # Updated initial baseline cosmological model if relevant
        # arguments specified.
        if baseline_model_kwargs is not None:
            self._set_baseline_model(**baseline_model_kwargs)

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
        with warnings.catch_warnings():
            # Suppress `mcfit` warning.
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
        with warnings.catch_warnings():
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

        # If `cosmo` is not `None`, `self._kernel` as A(k) (see docstring),
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
        Cosmological model used to produce the transfer function and
        power spectrum and to compute the linear growth rate.  If `None`
        (default) and not subsequently provided when calling some of the
        methods, non-Gaussianity modifications cannot be computed.  This
        can be subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    power_spectrum : callable or None, optional
        Linear matter power spectrum model at `redshift`.  When `cosmo`
        is provided, this is ignored; otherwise this cannot be `None`
        unless it is subsequently provided when calling some of the
        methods.  This can be subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    growth_rate : float or None, optional
        Linear growth rate at `redshift`.  If `None` (default), this is
        set by `cosmo` if it is provided; otherwise this is set to 0.
        This can be subsequently updated when calling
        :meth:`~.two_point_correlator` or :meth:`~.correlator_matrix`.
    couplings : :class:`~.couplings.Couplings` *or None, optional*
        Compiled coupling coefficients for the same cosmological model
        `cosmo`.  If `None` (default), this is compiled from
        `survey_specs` and `cosmo_specs` if either is provided; otherwise
        couplings are assumed to be trivial (i.e. reduced to Kronecker
        deltas).
    survey_specs : dict{str: callable or None} or None, optional
        Survey specification functions to be passed as `survey_specs` to
        :class:`~harmonia.reader.couplings.Couplings`.  Also used in
        shot noise calculations.
    cosmo_specs : dict{str: callable or None} or None, optional
        Cosmological specification functions to be passed as `cosmo_specs`
        to :class:`~harmonia.reader.couplings.Couplings`.  If this is set,
        its attributes must be consistent with the baseline model `cosmo`.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

    Attributes
    ----------
    couplings : :class:`nbodykit.cosmology.cosmology.Cosmology`
        Spherical coupling coefficients.
    cosmo : :class:`nbodykit.cosmology.cosmology.Cosmology` or None
        Cosmological model.
    matter_power_spectrum : callable
        Matter power spectrum at `redshift`.
    growth_rate : float
        Linear growth rate at `redshift`.
    attrs : dict
        Any other attributes inherited from input parameters.

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

        self._survey_specs = survey_specs

        if couplings is not None:
            self.couplings = couplings
        elif survey_specs is not None or cosmo_specs is not None:
            self.couplings = Couplings(
                self._disc, survey_specs=survey_specs, cosmo_specs=cosmo_specs
            )
        else:
            self.couplings = None

        # Set a baseline model.  May be subsequently updated.  Also
        # pre-compute quantities.
        self._set_baseline_model(
            cosmo, power_spectrum, growth_rate, cosmo_specs,
            update_couplings=False
        )

        # Compile powers of fixed modes, cosmology-independent angular
        # coupling sums and shot noise levels as directories to reduce
        # computational redundancy.
        self._angular_sums = self._compile_angular_sums()
        self._shot_noise_levels = self._compile_shot_noise_levels()

    def two_point_correlator(self, mu, nu, b_1, f_nl=None, nbar=None,
                             contrast=None, tracer_p=1.,
                             baseline_model_kwargs=None):
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
        baseline_model_kwargs : dict or None, optional
            Parameters `cosmo`, `power_spectrum`, `growth_rate` and
            `cosmo_specs` passed as keyword arguments to reset the
            baseline cosmological model.

        Returns
        -------
        complex
            Cosmological signal 2-point function value for given triplet
            indices.  If `cosmo_specs` is passed, radial and RSD couplings
            will be updated.

        """
        if baseline_model_kwargs is not None:
            self._set_baseline_model(
                update_couplings=True, **baseline_model_kwargs
            )

        # Aliases for readability and brevity.
        kappa, p_k, f = \
            self._disc.normalisations, self._mode_powers, self.growth_rate

        # pylint: disable=protected-access
        Phi = self.couplings._couplings['radial']
        Upsilon = self.couplings._couplings['rsd']

        # Summing contributions over degree index (say, ell_sigma).
        signal = 0.
        for ell_sigma, nmax_sigma in \
                zip(self._disc.degrees, self._disc.depths):

            b_k = b_1 * np.ones(nmax_sigma)
            if f_nl:
                b_k += f_nl * (b_1 - tracer_p) * np.asarray([
                    self._mode_modification[ell_sigma, n_sigma]
                    for n_sigma in range(1, nmax_sigma + 1)
                ])

            if f:
                radial_sum = np.sum([
                    Phi[mu, (ell_sigma, n_sigma)]
                    * Phi[nu, (ell_sigma, n_sigma)]
                    * b_k**2 * p_k[ell_sigma, n_sigma]
                    / kappa[ell_sigma, n_sigma]
                    for n_sigma in range(1, nmax_sigma + 1)
                ])
            else:
                radial_sum = np.sum([
                    (
                        b_k * Phi[mu, (ell_sigma, n_sigma)]
                        + f * Upsilon[mu, (ell_sigma, n_sigma)]
                    ) * (
                        b_k * Phi[nu, (ell_sigma, n_sigma)]
                        + f * Upsilon[nu, (ell_sigma, n_sigma)]
                    )
                    * p_k[ell_sigma, n_sigma] / kappa[ell_sigma, n_sigma]
                    for n_sigma in range(1, nmax_sigma + 1)
                ])

            signal += self._angular_sums[mu, nu][ell_sigma] * radial_sum

        if nbar is not None:
            alpha = 0. if contrast is None else 1 / contrast
            shot_noise = (1 + alpha) / nbar * self._shot_noise_levels[mu, nu]
            return signal + shot_noise

        return signal

    def correlator_matrix(self, pivot, b_1=None, f_nl=None, nbar=None,
                          contrast=None, tracer_p=1., radialise=False,
                          shot_noise_only=False, baseline_model_kwargs=None):
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
        radialise : bool, optional
            If `True`, return the diagonal matrix assuming mutual mode
            independence (only valid for no survey or cosmological
            specifications, i.e. trivial coupling coefficients.)
        baseline_model_kwargs : dict or None, optional
            Parameters `cosmo`, `power_spectrum`, `growth_rate` and
            `cosmo_specs` passed as keyword arguments to reset the
            baseline cosmological model.  If `cosmo_specs` is passed,
            radial and RSD couplings will be updated.

        Returns
        -------
        complex :class:`numpy.ndarray`
            Two-point correlator matrix vectorised by the given pivot.

        See Also
        --------
        :class:`~harmonia.algorithms.arrays.SphericalArray`

        """
        if baseline_model_kwargs is not None:
            self._set_baseline_model(
                update_couplings=True, **baseline_model_kwargs
            )

        index_array = SphericalArray(self._disc)
        if pivot == 'natural':
            index_vector = [
                tuple(index)
                for index in index_array.array['index']
            ]
        elif pivot == 'spectral':
            index_vector = [
                tuple(index)
                for index in index_array.array['index'][
                    np.argsort(index_array['wavenumber'])
                ]
            ]

        dim = len(index_vector)

        ## Radialisation: Only return diagonal matrix.

        if radialise:
            # Shot noise contribution.
            if nbar is None:
                shot_noise_diag = np.zeros(dim, dtype=complex)
            else:
                alpha = 0. if contrast is None else 1 / contrast
                shot_noise_diag = (1 + alpha) / nbar * np.asarray([
                    self._shot_noise_levels[mu, mu]
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

        ## Full: Correlator matrix.

        # Shot noise contribution.
        shot_noise_mat = np.zeros((dim,) * 2, dtype=complex)

        if nbar is not None:
            alpha = 0. if contrast is None else 1 / contrast
            shot_noise_amplitude = (1 + alpha) / nbar
            for row_idx in range(dim):
                for col_idx in range(row_idx + 1):
                    mu, nu = index_vector[row_idx], index_vector[col_idx]
                    shot_noise_mat[row_idx, col_idx] = \
                        shot_noise_amplitude * self._shot_noise_levels[mu, nu]

        if shot_noise_only:
            return shot_noise_mat

        # Signal contribution.
        signal_mat = np.zeros((dim,) * 2, dtype=complex)

        for row_idx in range(dim):
            for col_idx in range(row_idx + 1):
                mu, nu = index_vector[row_idx], index_vector[col_idx]
                signal_mat[row_idx, col_idx] = self.two_point_correlator(
                    mu, nu, b_1=b_1, f_nl=f_nl, tracer_p=tracer_p
                )

        # Use Hermitian property to fill in the strictly upper triangular part.
        two_point_corr_mat = signal_mat + shot_noise_mat
        two_point_corr_mat[np.triu_indices(dim, k=1)] = np.conj(
            np.transpose(two_point_corr_mat)
        )[np.triu_indices(dim, k=1)]

        return two_point_corr_mat

    def _set_baseline_model(self, cosmo=None, power_spectrum=None,
                            growth_rate=None, cosmo_specs=None,
                            update_couplings=False):

        # Only necessary to update radial and RSD couplings when `cosmo_specs`
        # is specified.  Previously compute angular couplings are attached.
        if cosmo_specs is not None and update_couplings:
            # pylint: disable=protected-access
            couplings = Couplings(
                self._disc,
                survey_specs=self._survey_specs, cosmo_specs=cosmo_specs,
                external_angular_couplings=self.couplings._couplings['angular']
            )
            couplings.compile_couplings()
        self._cosmo_specs = cosmo_specs

        # If `cosmo` is not `None`, `self._kernel` as A(k) (see docstring),
        # `self._mode_modification` is its evaluation at wavenumbers.
        # `self._mode_powers` is the evaluated matter power spectrum.
        self.cosmo = cosmo
        if self.cosmo is None:
            self.matter_power_spectrum = power_spectrum
            self.growth_rate = growth_rate or 0.
            self._kernel = None
            self._mode_modification = None
        else:
            self.matter_power_spectrum = cosmology.LinearPower(
                self.cosmo, redshift=self._z, transfer='CLASS'
            )
            self.growth_rate = growth_rate if growth_rate is not None \
                else self.cosmo.scale_independent_growth_rate(self._z)
            self._kernel = scale_dependence_modification(self.cosmo, self._z)
            self._mode_modification = {
                index: self._kernel(mode)
                for index, mode in self._disc.wavenumbers.items()
            }

        # (Re-)set pre-computed quantities.
        try:
            self._mode_powers = {
                index: self.matter_power_spectrum(mode)
                for index, mode in self._disc.wavenumbers.items()
            }
        except TypeError:
            warnings.warn(
                "Cannot evaluate :attr:`matter_power_spectrum`. "
                "No baseline model is current set."
            )

    def _compile_angular_sums(self):

        # Compiles all index pairs of the form (mu_ell, mu_m, nu_ell, nu_m).
        index_pair_vector = list(product(
            *(self._gen_operable_indices(subtype='angular'), ) * 2
        ))

        # Compile, for each index pair of the form above, all its angular sums
        # indexed by a new degree index (say, ell_sigma) of the form:
        # sum_{m_ell} (mu_ell, mu_m, ell_sigma, m_ell)
        # * conj((nu_ell, nu_m, ell_sigma, m_ell)).
        angular_sums_by_index_pair = mpi_compute(
            index_pair_vector, self._compile_angular_sums_by_index_pair,
            comm=self.comm, logger=self.logger,
            process_name="angular sum compilation"
        )

        # Store compiled results in a directory, where each key is an index
        # pair of the form above, and each entry is a dictionary of angular
        # sums indexed by the new degree index (ell_sigma).
        angular_sums = {}
        for index_pair, angular_sums_for_index_pair \
                in zip(index_pair_vector, angular_sums_by_index_pair):
            angular_sums.update({
                tuple(index_pair): angular_sums_for_index_pair
            })

        return angular_sums

    def _compile_shot_noise_levels(self):

        # Compiles all index pairs of the form (mu_ell, mu_m, nu_ell, nu_m).
        index_pair_vector = list(product(
            *(self._gen_operable_indices(), ) * 2
        ))

        # Compile, for each index pair of the form above, the shot noise
        # level of the form: M_{mu, nu} * radial integral [selection, weight].
        shot_noise_levels_by_index_pair = mpi_compute(
            index_pair_vector, self._compile_shot_noise_levels_by_index_pair,
            comm=self.comm, logger=self.logger,
            process_name="shot noise level compilation"
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

        return shot_noise_levels

    def _compile_angular_sums_by_index_pair(self, index_pair):

        mu, nu = index_pair

        # For each degree index (say ell_sigma), return sum_{m_sigma}
        # of M_{mu, sigma} * conj(M_{nu, sigma}).
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
        # sph_besselj(ell_mu, k_mu) * sph_besselj(ell_nu, k_nu).
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
                "Poorly determined shot noise integral "
                "for index pair {} and {}.".format(mu, nu),
                category=IntegrationWarning
            )

        return M_mu_nu * shot_noise_integral

    def _gen_operable_indices(self, subtype=None):

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

        return operable_indices
