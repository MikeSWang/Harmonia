r"""
Spherical Fourier-space model (:mod:`~harmonia.reader.spherical_model`)
===========================================================================

Compute Fourier-space :math:`n`-point functions in spherical basis.  The
index for each Fourier coefficient is a triplet :math:`(\ell_\mu, m_\mu,
n_\mu)`, i.e. ``(mu[0], mu[1], mu[2])``.

.. _degree-index-warning:

.. warning::

    In this module, all variables related to the discretised spectrum
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`, such as
    `wavenumbers` and `normalisations`, are assumed to start at spherical
    degree :math:`\ell = 0`.

Kernels
---------------------------------------------------------------------------

Coupling kernels are integrands without the coordinate Jacobian, which may
include the following factors: radial selection :math:`\phi(r)`, weight
:math:`w(r)` or its derivative, and angular mask
:math:`M(\hat{\mathbf{r}})`; linear growth rate normalised to linear bias
:math:`\beta(z) = f(z)/b(z)`, clustering evolution :math:`G(z) = b(z) D(z)`
where :math:`D(z)` is the linear growth factor, and the Alcock--Paczynski
distortion

.. math::

    \gamma(z) = \frac{\beta(z)}{\beta_0}
    \frac{\mathrm{d}\tilde{r}}{\mathrm{d}r} \,, \quad
    \text{with} \quad \beta_0 \equiv \beta(0) \,,

where :math:`\tilde{r} = \tilde{r}(z)` is the fiducial distance converted
from redshift rather than from the true comoving distance--redshift
correspondence :math:`z = z(r)`.

When using integration kernels that is a combination of functions of the
above, pass additional parameters not being directly integrated over by
redefining these functions with ``lambda``.

Couplings
---------------------------------------------------------------------------

Coupling coefficients are computed by integrating the angular, radial and
RSD coupling kernels

.. math::

   Y_{\ell_\mu m_\mu}^*(\hat{\mathbf{r}}) M(\hat{\mathbf{r}})
   Y_{\ell_\nu m_\nu}(\hat{\mathbf{r}}) \,, \\

   \kappa_{\ell_\nu n_\nu} w(\tilde{r})
   j_{\ell_\mu}(k_{\ell_\mu n_\mu} \tilde{r})
   j_{\ell_\nu}(k_{\ell_\nu n_\nu} r) G(z) \phi(r) \,, \\

   \frac{\kappa_{\ell_\nu n_\nu}}{k_{\ell_\nu n_\nu}}
   \frac{\mathrm{d}}{\mathrm{d}\tilde{r}}
   \Big[ w(\tilde{r}) j_{\ell_\mu}(k_{\ell_\mu n_\mu} \tilde{r}) \Big]
   j'_{\ell_\nu}(k_{\ell_\nu n_\nu} r) \gamma(z) G(z) \phi(r) \,,

over the spherical Lebesgue measure, where :math:`\{ k_{\ell n} \}` are the
discrete wavenumbers.

.. autosummary::

    Couplings

2-point functions
---------------------------------------------------------------------------

2-point functions are computed from couplings as a sum of the signal part

.. math::

    \left\langle \delta_\mu \delta_\nu \right\rangle = \sum_\sigma
    M_{\mu\sigma} M^*_{\nu\sigma} \left( \Phi_{\mu\sigma} +
    \beta_0 \Upsilon_{\mu\sigma} \right) \left( \Phi_{\nu\sigma} +
    \beta_0 \Upsilon_{\nu\sigma} \right) \kappa_\sigma^{-1} P(k_\sigma)

and the shot noise part

.. math::

    \left\langle \epsilon_\mu \epsilon_\nu \right\rangle =
    \frac{1}{\bar{n}} M_{\mu\nu} \int \mathrm{d}r r^2 (w^2\phi)(r)
    j_\mu(r) j_\nu(r) \,,

where :math:`M, \Phi, \Upsilon` are the angular, radial and RSD couplings
and :math:`\kappa` the normalisation coefficients (see also
:class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`), and
:math:`j_\mu(r) \equiv j_{\ell_\mu}(k_{\ell_\mu n_\mu} r)`.

.. autosummary::

    TwoPointFunction

|

"""
import logging
import warnings

import numpy as np

from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic
from harmonia.algorithms.integration import (
    angular_spherical_integral as ang_int,
    radial_spherical_integral as rad_int,
)
from harmonia.algorithms.morph import SphericalArray
from harmonia.collections.utils import mpi_compute


# KERNELS
# -------------------------------------------------------------------------

def _angular_kernel(theta, phi, mu, nu, mask=None):
    r"""Evaluate the angular coupling kernel.

    Parameters
    ----------
    theta, phi : float, array_like
        Angular coordinates :math:`(\theta, \phi)`.
    mu, nu : tuple or list of int
        Coefficient triplet index.
    mask : callable or None, optional
        Mask as a function of angular coordinates (default is `None`).  The
        arguments must be in the following order and range:
        :math:`0 \leqslant \theta \leqslant \pi`, :math:`0 \leqslant \phi
        \leqslant 2\pi`.

    Returns
    -------
    kernel : complex, array_like
        Angular coupling kernel value.

    """
    kernel = np.conj(spherical_harmonic(mu[0], mu[1], theta, phi)) \
        * spherical_harmonic(nu[0], nu[1], theta, phi)

    if hasattr(mask, '__callable__'):
        kernel *= mask(theta, phi)
    else:
        warnings.warn(
            "`mask` is None. Angular model evaluation may be redundant. ",
            RuntimeWarning,
        )

    return kernel


def _radial_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
                  evolution=None, r2z=None, z2chi=None):
    """Evaluate the radial coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple or list of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default
        is `None`).
    evolution : callable or None, optional
        Evolution as a function of redshift (default is `None`).
    r2z : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default is
        `None`).
    z2chi : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is
        `None`).

    Returns
    -------
    kernel : float, array_like
        Radial coupling kernel value.

    Raises
    ------
    ValueError
        If `r2z` is not callable when either `evolution` or `z2chi` is.

    """
    if not callable(z2chi):
        r_tilde = r
    else:
        if callable(r2z):
            r_tilde = z2chi(r2z(r))
        else:
            raise ValueError("`r2z` must be callable if `z2chi` is. ")

    kernel = spherical_besselj(mu[0], k_mu*r_tilde) \
        * spherical_besselj(nu[0], k_nu*r)

    if callable(selection):
        kernel *= selection(r)
    if callable(weight):
        kernel *= weight(r_tilde)
    if callable(evolution):
        if not callable(r2z):
            raise ValueError("`r2z` must be callable if `evolution` is. ")
        kernel *= evolution(r2z(r))

    return kernel


def _RSD_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
               weight_derivative=None, evolution=None, AP_distortion=None,
               r2z=None, z2chi=None):
    """Evaluate the RSD coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple or list of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default
        is `None`).
    weight_derivative : callable or None, optional
        Weight function derivative as a function of the radial coordinate
        (default is `None`).
    evolution, AP_distortion : callable or None, optional
        Evolution or AP distortion as a function of redshift (default is
        `None`).
    r2z : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default is
        `None`).
    z2chi : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is
        `None`).

    Returns
    -------
    kernel : float, array_like
        RSD coupling kernel value.

    Raises
    ------
    ValueError
        If `weight_derivative` is not callable when `weight` is.
    ValueError
        If `r2z` is not callable when one of `z2chi`, `evolution` and
        `AP_distortion` is,

    """
    if not callable(z2chi):
        r_tilde = r
    else:
        if callable(r2z):
            r_tilde = z2chi(r2z(r))
        else:
            raise ValueError("`r2z` must be callable if `z2chi` is. ")

    kernel = spherical_besselj(nu[0], k_nu*r, derivative=True)

    if callable(selection):
        kernel *= selection(r)

    if not callable(weight):
        kernel *= k_mu * spherical_besselj(mu[0], k_mu*r, derivative=True)
    else:
        if not callable(weight_derivative):
            raise ValueError(
                "`weight_derivative` must be callable if `weight` is. "
            )
        kernel *= weight_derivative(r_tilde) \
            * spherical_besselj(mu[0], k_mu*r_tilde) \
            + k_mu * weight(r_tilde) \
            * spherical_besselj(mu[0], k_mu*r_tilde, derivative=True)

    if callable(evolution):
        if not callable(r2z):
            raise ValueError("`r2z` must be callable if `evolution` is. ")
        kernel *= evolution(r2z(r))

    if callable(AP_distortion):
        if not callable(r2z):
            raise ValueError("`r2z` must be callable if `AP_distortion` is. ")
        kernel *= AP_distortion(r2z(r))

    return kernel


def _shot_noise_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None):
    """Evalaute the shot noise 2-point function kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple or list of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default
        is `None`).

    Returns
    -------
    kernel : float, array_like
        Shot noise 2-point function kernel value.

    """
    if selection is None and weight is None and mu[0] == nu[0]:
        warnings.warn(
            "Shot noise evaluation may be redundant. ",
            RuntimeWarning,
        )

    kernel = spherical_besselj(mu[0], k_mu*r) \
        * spherical_besselj(nu[0], k_nu*r)
    if callable(selection):
        kernel *= selection(r)
    if callable(weight):
        kernel *= weight(r)**2

    return kernel


# COUPLINGS
# -----------------------------------------------------------------------------

class Couplings:
    """Compute angular, radial and RSD coupling coefficients for given
    survey and cosmological specifications.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    survey_specs : dict of callable or None, or None, optional
        Survey specification functions accessed with the following
        mandatory keys: ``'mask'`` for angular mask, and ``'selection'``
        and ``'weight'`` for radial selection and weighting,
        ``'weight_derivative'`` for the derivative function of radial
        weighting.  Default is `None`.
    cosmo_specs : dict of callable or None, or None, optional
        Cosmological specification functions accessed with the following
        mandatory keys: ``'r2z'`` for cosmological comoving
        distance-to-redshift conversion, ``'z2chi'`` for fiducial
        redshift-to-comoving distance conversion, ``'evolution'`` for
        clustering evolution, and ``'distorion_AP'`` for AP distortion.
        Default is `None`.

    Attributes
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    mask : callable or None
        Angular mask function.
    selection : callable or None
        Selection function of the radial coordinate.
    weight : callable or None
        Weight function of the radial coordinate.
    weight_derivative : callable or None
        Weight derivative function of the radial coordinate.
    r2z : callable or None
        Cosmological comoving distance-to-redshift conversion.
    z2chi : callable or None
        Fiducial comoving redshift-to-distance conversion.
    evolution : callable or None
        Clustering evolution function of redshift.
    distotion_AP : callable or None
        AP distortion function of redshift.

    Raises
    ------
    KeyError
        If `survey_specs` and `cosmo_specs` are passed as dictionaries but
        one of the keys corresponding to a required function is missing.
    TypeError
        If `survey_specs` and `cosmo_specs` are passed as dictionaries but
        one of the values as a required function is neither `None` nor
        callable.

    """

    _logger = logging.getLogger("Couplings")

    _all_specs_attr = {
        'survey_specs': ('mask', 'selection', 'weight', 'weight_derivative'),
        'cosmo_specs': ('r2z', 'z2chi', 'evolution', 'distotion_AP'),
    }

    def __init__(self, disc, survey_specs=None, cosmo_specs=None):

        self.disc = disc

        for specs_name, specs_attrs in self._all_specs_attr.items():
            specs_var_str = "`" + specs_name + "`"
            specs = locals()[specs_name]
            if isinstance(specs, dict):
                try:
                    for attr in specs_attrs:
                        setattr(self, attr, specs[attr])
                        attr_val = getattr(self, attr)
                        if not callable(attr_val) \
                                and attr_val is not None:
                            raise TypeError(
                                specs_var_str +
                                f" {attr} value must be None or callable. "
                            )
                except KeyError as missing_key:
                    raise KeyError(
                        specs_var_str + f" key {missing_key} is missing. "
                    )

    def coupling_coefficient(self, mu, nu, coupling_type):
        r"""Evaluate couplings at specified indices.

        When there is no angular masking (i.e. `mask` is `None`), the
        coupling coefficients reduce to :math:`M_{\mu\nu} =
        \delta_{\mu\nu}`.  When there is no angular masking or clustering
        evolution, if radial selection and weight are both absent and the
        distance--redshift conversion is the cosmological one (i.e. none of
        `mask`, `selection`, `weight`, `evolution`, `r2z` and `z2chi` is
        set), the coupling coefficients reduce to :math:`\Phi_{\mu\nu} =
        \delta_{\mu\nu}`.

        Parameters
        ----------
        mu, nu : tuple or list [of length 3] of int
            Coefficient triplet index.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.

        Returns
        -------
        coupling_coeff : float or complex, array_like
            Coupling coefficient of given type for specified indices.

        Raises
        ------
        ValueError
            If `coupling_type` does not correspond to a valid kernel.

        """
        _info_msg = "{} coupling for: {} and {}".format(
            coupling_type.replace("'", ""),
            str(mu).replace("'", ""),
            str(nu).replace("'", ""),
        )
        self._logger.debug("Computing %s", _info_msg)

        if coupling_type.lower.startswith('a'):
            trivial_case = not hasattr(self.mask, '__call__')
            if trivial_case:
                if mu[0] == nu[0] and mu[1] == nu[1]:
                    return 1. + 0.j
                return 0. + 0.j

            def _ang_kernel(theta, phi):
                return _angular_kernel(theta, phi, mu, nu, mask=self.mask)

            coupling_coeff = ang_int(_ang_kernel)
            self._logger.debug("Computed %s", _info_msg)

            return coupling_coeff

        rmax = self.disc.attrs['boundary_radius']

        ell_mu, n_mu = mu[0], mu[-1]
        ell_nu, n_nu = nu[0], nu[-1]

        k_mu = self.disc.wavenumbers[ell_mu][n_mu-1]
        k_nu = self.disc.wavenumbers[ell_nu][n_nu-1]
        kappa_nu = self.disc.normalisations[ell_nu][n_nu-1]

        if coupling_type.lower.startswith('rad'):
            attrs = ['selection', 'weight', 'evolution', 'r2z', 'z2chi']
            funcs = {attr: getattr(self, attr) for attr in attrs}

            trivial_case = not any(
                [callable(func) for attr, func in funcs.items()],
            )
            if trivial_case:
                if mu[0] == nu[0]:
                    return float(mu[-1] == nu[-1])

            coupling_coeff = kappa_nu * rad_int(
                lambda r: _radial_kernel(r, mu, nu, k_mu, k_nu, **funcs),
                rmax,
            )
            self._logger.debug("Computed %s", _info_msg)

            return coupling_coeff

        if coupling_type.lower.startswith('rsd'):
            attrs = [
                'selection',
                'weight',
                'weight_derivative',
                'evolution',
                'AP_distortion',
                'r2z',
                'z2chi',
                ]
            funcs = {attr: getattr(self, attr) for attr in attrs}

            coupling_coeff = kappa_nu / k_nu * rad_int(
                lambda r: _RSD_kernel(r, mu, nu, k_mu, k_nu, **funcs),
                rmax,
            )
            self._logger.debug("Computed %s", _info_msg)

            return coupling_coeff

        raise ValueError(
            "`coupling_type` can only be: 'angular', 'radial' or 'RSD'. "
        )

    def couplings_single_component(self, mu, coupling_type):
        r"""Compute coupling coefficients with the first triplet index
        fixed.

        Note
        ----
        This function computes coupling coefficients of the form
        :math:`C_{a_\mu b_\mu a_\sigma b_\sigma}` where the triplet index
        :math:`\mu` is fixed, and compiles their values as a vector by
        iterating through the triplet index :math:`\sigma`.  For example,

        ::

            Couplings.compile_over_index(mu, 'angular')

        returns the quantity

        .. math::

            M_{\ell_\mu m_\mu \cdot \cdot} = {
                \left\lbrace
                   M_{\ell_\mu m_\mu \ell_\sigma m_\sigma}
                   \,\middle\vert\,
                   m_\sigma = -\ell_\sigma, \dots, \ell_\sigma
                \right\rbrace
            }_{\sigma} \,.

        Parameters
        ----------
        mu : tuple or list of int
            Fixed triplet index.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.

        Returns
        -------
        couplings_component_array : complex or float, array_like
            Array of coupling coefficients with first triplet index fixed.

        Raises
        ------
        ValueError
            If `coupling_type` does not correspond to a valid kernel.

        """
        if coupling_type.lower().startswith('a'):
            sigma_gen = lambda ell: [
                (ell, m, None) for m in range(-ell, ell+1)
            ]
        elif coupling_type.lower().startswith(('rad', 'rsd')):
            sigma_gen = lambda ell: [
                (ell, None, n) for n in range(1, self.disc.depths[ell]+1)
            ]
        else:
            raise ValueError(
                "`coupling_type` can only be: 'angular', 'radial' or 'RSD'. "
            )

        couplings_component_array = []
        for ell in enumerate(self.disc.degrees):
            sigma_component = [
                self.coupling_coefficient(mu, sigma, coupling_type)
                for sigma in sigma_gen(ell)
            ]
            couplings_component_array.append(np.array(sigma_component))

        return couplings_component_array

    def compile_couplings_vector(self, pivot, coupling_type, comm=None):
        r"""Compile all coupling coefficients of a given type as a vector
        iterated through the first triplet index ordered as specified.

        This returns a list whose entries, ordered and indexed by ``mu``,
        are ``Couplings.compile_over_index(mu, coupling_type)``.

        Parameters
        ----------
        pivot : {'natural', 'lmn', 'lnm', 'k'}
            Vectorisation order.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.
        comm : :class:`mpi4py.MPI.Comm` or None, optional
            MPI communicator.  If `None` (default), no multiprocessing
            is invoked.

        Returns
        -------
        couplings_vectorised_arrays : complex or float, array_like
            Vector of coupling coefficients over the first triplet index.

        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        index_vector = SphericalArray.build(disc=self.disc)\
            .unfold(pivot, return_only='index')

        if comm is not None:
            process_index = lambda mu: self.couplings_single_component(
                mu,
                coupling_type=coupling_type,
            )
            couplings_vectorised_arrays = mpi_compute(
                index_vector,
                process_index,
                comm
            )
        else:
            couplings_vectorised_arrays = [
                self.couplings_single_component(
                    mu,
                    coupling_type=coupling_type,
                )
                for mu in index_vector
            ]

        return couplings_vectorised_arrays


# 2-Point Correlators
# -----------------------------------------------------------------------------

def two_point_shot_noise(mu, nu, nbar, disc, M_mu_nu, selection=None,
                         weight=None):
    r"""Compute the shot noise 2-point function.

    Parameters
    ----------
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    M_mu_nu : complex
        Angular mask coupling coefficients :math:`M_{\mu\nu}`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default
        is `None`).

    Returns
    -------
    shot_noise : complex, array_like
        Shot noise 2-point function value for given indices.

    """
    if np.allclose(M_mu_nu, 0.):
        return 0.

    rmax = disc.attrs['boundary_radius']

    ell_mu, n_mu = mu[0], mu[-1]
    ell_nu, n_nu = nu[0], nu[-1]

    u_mu = disc.roots[ell_mu][n_mu-1]
    k_mu = disc.wavenumbers[ell_mu][n_mu-1]
    k_nu = disc.wavenumbers[ell_nu][n_nu-1]

    if not callable(selection) and not callable(weight) and ell_mu == ell_nu:
        if n_mu == n_nu:
            shot_noise = rmax**3 * spherical_besselj(ell_mu+1, u_mu)**2 / 2
        else:
            shot_noise = 0.
    else:
        args = mu, nu, k_mu, k_nu
        kwargs = dict(selection=selection, weight=weight)
        shot_noise = rad_int(
            lambda r: _shot_noise_kernel(r, *args, **kwargs),
            rmax,
        )

    shot_noise *= M_mu_nu / nbar

    return shot_noise


class TwoPointFunction(Couplings):
    """Compute 2-point function values for given survey and cosmological
    specifications from a power spectrum model and RSD growth rate.

    Parameters
    ----------
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    power_spectrum : callable
        Linear galaxy-clustering power spectrum model (in cubic Mpc/h).
    beta_0 : float
        Linear growth rate over bias :math:`\beta_0` at the current epoch.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    survey_specs : dict of callable or None, or None, optional
        Survey specification functions accessed with the following
        mandatory keys: ``'mask'`` for angular mask, and ``'selection'``
        and ``'weight'`` for radial selection and weighting,
        ``'weight_derivative'`` for the derivative function of radial
        weighting.  Default is `None`.
    cosmo_specs : dict of callable or None, or None, optional
        Cosmological specification functions accessed with the following
        mandatory keys: ``'r2z'`` for cosmological comoving
        distance-to-redshift conversion, ``'z2chi'`` for fiducial
        redshift-to-comoving distance conversion, ``'evolution'`` for
        clustering evolution, and ``'distorion_AP'`` for AP distortion.
        Default is `None`.
    comm : :class:`mpi4py.MPI.Comm` or None, optional
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

    Attributes
    ----------
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    power_spectrum : callable
        Linear galaxy-clustering power spectrum model (in cubic Mpc/h).
    beta_0 : float
        Linear growth rate over bias :math:`\beta_0` at the current epoch.
    comm : :class:`mpi4py.MPI.Comm` or None, optional
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

    Raises
    ------
    KeyError
        If `survey_specs` and `cosmo_specs` are passed as dictionaries but
        one of the keys corresponding to a required function is missing.
    TypeError
        If `survey_specs` and `cosmo_specs` are passed as dictionaries but
        one of the values as a required function is neither `None` nor
        callable.

    """

    _logger = logging.getLogger("TwoPointFunction")

    def __init__(self, nbar, power_spectrum, beta_0, disc, survey_specs=None,
                 cosmo_specs=None, comm=None):

        self.nbar = nbar
        self.power_spectrum = power_spectrum
        self.beta_0 = beta_0
        self._natural_couplings = None

        super().__init__(
            disc,
            survey_specs=survey_specs,
            cosmo_specs=cosmo_specs,
        )

    def _compute_natural_couplings(self, comm=None):

        if self._natural_couplings is not None:
            return self._natural_couplings

        self._natural_couplings = dict.fromkeys(['angular', 'radial', 'RSD'])
        for coupling_type in self._natural_couplings:
            self._natural_couplings[coupling_type] = super()\
                .compile_couplings_vector(
                    'natural',
                    coupling_type,
                    comm=comm,
                )
        self._logger.info("Natural coupling coefficients computed. ")

        return self._natural_couplings

    def two_point_signal_value(self, mu, nu, comm=None):
        r"""Compute the 2-point function signal for given indices.

        Parameters
        ----------
        mu, nu :  tuple or list of int
            Coefficient triplet index.

        Returns
        -------
        signal_value : complex
            Cosmological signal 2-point function value for given triple
            indices.

        """
        k, kappa = disc.wavenumbers, disc.normalisations

        M_mu, M_nu = mu_couplings['angular'], nu_couplings['angular']
        Phi_mu, Phi_nu = mu_couplings['radial'], nu_couplings['radial']
        Upsilon_mu, Upsilon_nu = mu_couplings['RSD'], nu_couplings['RSD']

        signal = 0
        for ell_idx, ell in enumerate(disc.degrees):
            M_mu_, M_nu_ = M_mu[ell_idx], M_nu[ell_idx]
            Phi_mu_, Phi_nu_ = Phi_mu[ell_idx], Phi_nu[ell_idx]
            Upsilon_mu_, Upsilon_nu_ = Upsilon_mu[ell_idx], Upsilon_nu[ell_idx]
            k_ell, kappa_ell = k[ell_idx], kappa[ell_idx]
            angular_sum = np.sum(
                [
                    M_mu_[m_idx] * np.conj(M_nu_[m_idx])
                    for m_idx in range(0, 2*ell+1)
                ]
            )
            radial_sum = np.sum(
                [
                    (Phi_mu_[n_idx] + beta_0 * Upsilon_mu_[n_idx])
                    * (Phi_nu_[n_idx] + beta_0 * Upsilon_nu_[n_idx])
                    * power_spectrum(k_ell[n_idx]) / kappa_ell[n_idx]
                    for n_idx in range(disc.depths[ell_idx])
                ]
            )
            signal += angular_sum * radial_sum

        return signal

    def two_point_shot_noise_value(self, mu, nu, comm=None):
        """Compute


        """
        pass