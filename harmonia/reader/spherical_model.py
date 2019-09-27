r"""
Spherical Fourier-space model (:mod:`~harmonia.reader.spherical_model`)
===============================================================================

Compute Fourier-space :math:`n`-point functions in spherical basis.  The index
for each Fourier coefficient is a triplet :math:`(\ell_\mu, m_\mu, n_\mu)`,
i.e. ``(mu[0], mu[1], mu[2])``.

.. _degree-index-warning:

.. topic:: Caution

    In this module, all variables related to the discretised spectrum
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`, such as
    `wavenumbers` and `normalisation`, are assumed to be in the natural
    structure starting at spherical degree :math:`\ell = 0` (see
    :class:`~harmonia.algorithms.morph.SphericalArray`).  In the future, these
    variables may be changed to :obj:`dict` without assuming this
    correspondence, and relevant :math:`\ell`-modes are accessed through keys.

Kernels
-------------------------------------------------------------------------------

Coupling kernels are integrands without the coordinate Jacobian, which may
include the following factors: radial selection :math:`\phi(r)`, weight
:math:`w(r)`, and angular mask :math:`M(\hat{\mathbf{r}})`; linear growth rate
normalised to linear bias :math:`\beta(z) = f(z)/b(z)`, clustering evolution
:math:`G(z) = b(z) D(z)` where :math:`D(z)` is the linear growth factor, and
the Alcock--Paczynski distortion

.. math::

    \gamma(z) = \frac{\beta(z)}{\beta_0}
    \frac{\mathrm{d}\tilde{r}}{\mathrm{d}r} \,, \quad
    \text{with} \quad \beta_0 \equiv \beta(0) \,.

When using integration kernels that is a combination of functions such as
weight, selection, mask and evolution etc., pass additional parameters not
being integrated over by redefining these functions with :obj:`lambda`.

Couplings
-------------------------------------------------------------------------------

Coupling coefficients are computed by integrating the angular, radial and RSD
coupling kernels

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

over the spherical Lebesgue measure,  where :math:`\tilde{r}` is the distance
converted in a fiducial cosmological model rather than from the true comoving
distance--redshift correspondence, and :math:`\{ k_{\ell n} \}` are the
discrete wavenumbers.

.. autosummary::

    Couplings

2-point functions
-------------------------------------------------------------------------------

2-point functions are computed from couplings as a sum of the signal part

.. math::

    \left\langle \delta_\mu \delta_\nu \right\rangle = \sum_\sigma
    M_{\mu\sigma} M^*_{\nu\sigma} \left( \Phi_{\mu\sigma} +
    \beta_0 \Upsilon_{\mu\sigma} \right) \left( \Phi_{\nu\sigma} +
    \beta_0 \Upsilon_{\nu\sigma} \right) \kappa_\sigma^{-1} P(k_\sigma)

and the shot noise part

.. math::

    \left\langle \epsilon_\mu \epsilon_\nu \right\rangle = \frac{1}{\bar{n}}
    M_{\mu\nu} \int \mathrm{d}r r^2 (w^2\phi)(r) j_\mu(r) j_\nu(r) \,,

where :math:`M, \Phi, \Upsilon` are the angular, radial and RSD couplings and
:math:`\kappa` the normalisation coefficients (see also
:class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`), and
:math:`j_\mu(r) \equiv j_{\ell_\mu}(k_{\ell_\mu n_\mu} r)`.

.. autosummary::

    two_point_signal
    two_point_shot_noise

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


# KERNELS
# -----------------------------------------------------------------------------

def _angular_kernel(theta, phi, mu, nu, mask=None):
    r"""Evaluate the angular coupling kernel.

    Parameters
    ----------
    theta, phi : float, array_like
        Angular coordinates :math:`(\theta, \phi)`.
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    mask : callable or None, optional
        Mask as a function of angular coordinates (default is `None`).

    Returns
    -------
    kernel : complex, array_like
        Angular coupling kernel value.

    Notes
    -----
    The first two positional arguments of `mask` must be in radians and in
    the following order and range: ``0 <= theta <= np.pi``,
    ``0 <= phi <= 2*np.pi``.

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
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default is
        `None`).
    evolution : callable or None, optional
        Evolution as a function of redshift (default is `None`).
    r2z : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default is
        `None`).
    z2chi : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is `None`).

    Returns
    -------
    kernel : float, array_like
        Radial coupling kernel value.

    Raises
    ------
    ValueError
        If `r2z` is not callable when either `evolution` or `z2chi` is.

    """
    if not hasattr(z2chi, '__call__'):
        r_tilde = r
    else:
        if hasattr(r2z, '__call__'):
            r_tilde = z2chi(r2z(r))
        else:
            raise ValueError("`r2z` must be callable if `z2chi` is. ")

    kernel = spherical_besselj(mu[0], k_mu*r_tilde) \
        * spherical_besselj(nu[0], k_nu*r)

    if hasattr(selection, '__call__'):
        kernel *= selection(r)
    if hasattr(weight, '__call__'):
        kernel *= weight(r_tilde)
    if hasattr(evolution, '__call__'):
        if not hasattr(r2z, '__call__'):
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
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default is
        `None`).
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
        Fiducial comoving redshift-to-distance conversion (default is `None`).

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
    if not hasattr(z2chi, '__call__'):
        r_tilde = r
    else:
        if hasattr(r2z, '__call__'):
            r_tilde = z2chi(r2z(r))
        else:
            raise ValueError("`r2z` must be callable if `z2chi` is. ")

    kernel = spherical_besselj(nu[0], k_nu*r, derivative=True)

    if hasattr(selection, '__call__'):
        kernel *= selection(r)

    if not hasattr(weight, '__call__'):
        kernel *= k_mu * spherical_besselj(mu[0], k_mu*r, derivative=True)
    else:
        if not hasattr(weight_derivative, '__call__'):
            raise ValueError(
                "`weight_derivative` must be callable if `weight` is. "
            )
        kernel *= weight_derivative(r_tilde) \
            * spherical_besselj(mu[0], k_mu*r_tilde) \
            + k_mu * weight(r_tilde) \
            * spherical_besselj(mu[0], k_mu*r_tilde, derivative=True)

    if hasattr(evolution, '__call__'):
        if not hasattr(r2z, '__call__'):
            raise ValueError("`r2z` must be callable if `evolution` is. ")
        kernel *= evolution(r2z(r))

    if hasattr(AP_distortion, '__call__'):
        if not hasattr(r2z, '__call__'):
            raise ValueError("`r2z` must be callable if `AP_distortion` is. ")
        kernel *= AP_distortion(r2z(r))

    return kernel


def _shot_noise_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None):
    """Evalaute the shot noise 2-point function kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default is
        `None`).

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
    if hasattr(selection, '__call__'):
        kernel *= selection(r)
    if hasattr(weight, '__call__'):
        kernel *= weight(r)**2

    return kernel


# COUPLINGS
# -----------------------------------------------------------------------------

class Couplings:
    r"""Compute angular, radial and RSD coupling coefficients for given survey
    and cosmological specifications.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    survey_specs : dict of callable or None, or None, optional
        Survey specification functions accessed with the following mandatory
        keys: ``'mask'`` for angular mask, and ``'selection'`` and ``'weight'``
        for radial selection and weighting, ``'weight_derivative'`` for the
        derivative function of radial weighting.  Default is `None`.
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
        If `survey_specs` and `cosmo_specs` are passed as dictionaries but one
        of the keys corresponding to a required function is missing.
    TypeError
        If `survey_specs` and `cosmo_specs` are passed as dictionaries but one
        of the values as a required function is neither `None` nor callable.

    """

    _logger = logging.getLogger("Couplings")

    _all_specs_attr = {
        'survey_specs': ('mask', 'selection', 'weight', 'weight_derivative'),
        'cosmo_specs': ('r2z', 'z2chi', 'evolution', 'distotion_AP'),
    }

    def __init__(self, disc=None, survey_specs=None, cosmo_specs=None):

        self.disc = disc

        for specs_name, specs_attrs in self._all_specs_attr.items():
            specs_var_str = "`" + specs_name + "`"
            specs = locals()[specs_name]
            if isinstance(specs, dict):
                try:
                    for attr in specs_attrs:
                        setattr(self, attr, specs[attr])
                        attr_val = getattr(self, attr)
                        if not hasattr(attr_val, '__call__') \
                                and attr_val is not None:
                            raise TypeError(
                                specs_var_str +
                                f" {attr} value must be None or callable. "
                            )
                except KeyError as missing_key:
                    raise KeyError(
                        specs_var_str + f" key {missing_key} is missing. "
                    )

    def __call__(self, mu, nu, coupling_type):
        """Evaluate couplings at specified indices.

        Paramaters
        ----------
        mu, nu : tuple or list [of length 3] of int
            Coefficient triplet index.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.

        Returns
        -------
        float or complex, array_like
            Coupling coefficient of given type for specified indices.

        Raises
        ------
        ValueError
            If `coupling_type` does not correspond to a valid kernel.

        Notes
        -----
        When there is no angular masking (i.e. `mask` is `None`), the coupling
        coefficients reduce to :math:`M_{\mu\nu} = \delta_{\mu\nu}`.  When
        there is no angular masking or clustering evolution, if radial
        selection and weight are both absent and the distance--redshift
        conversion is the cosmological one (i.e. none of `mask`, `selection`,
        `weight`, `evolution`, `r2z` and `z2chi` is set), the coupling
        coefficients reduce to :math:`\Phi_{\mu\nu} = \delta_{\mu\nu}`.

        """
        if coupling_type.lower.startswith('a'):
            trivial_case = not hasattr(self.mask, '__call__')
            if trivial_case:
                if mu[0] == nu[0] and mu[1] == nu[1]:
                    return 1. + 0.j
                return 0. + 0.j

            def _ang_kernel(theta, phi):
                return _angular_kernel(theta, phi, mu, nu, mask=self.mask)

            return ang_int(_ang_kernel)

        rmax = self.disc.attrs['boundary_radius']

        ell_mu, n_mu = mu[0], mu[-1]
        ell_nu, n_nu = nu[0], nu[-1]

        k_mu = self.disc.wavenumbers[ell_mu][n_mu-1]
        k_nu = self.disc.wavenumbers[ell_nu][n_nu-1]
        kappa_nu = self.disc.normalisation[ell_nu][n_nu-1]

        if coupling_type.lower.startswith('rad'):
            attrs = ['selection', 'weight', 'evolution', 'r2z', 'z2chi']
            funcs = {attr: getattr(self, attr) for attr in attrs}

            trivial_case = not any(
                [hasattr(func, '__call__') for attr, func in funcs.items()],
            )
            if trivial_case:
                if mu[0] == nu[0]:
                    return float(mu[-1] == nu[-1])

            return kappa_nu * rad_int(
                lambda r: _radial_kernel(r, mu, nu, k_mu, k_nu, **funcs),
                rmax,
            )

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

            return kappa_nu / k_nu * rad_int(
                lambda r: _RSD_kernel(r, mu, nu, k_mu, k_nu, **funcs),
                rmax,
            )

        raise ValueError(
            "`coupling_type` can only be: 'angular', 'radial' or 'RSD'. "
        )

    def compile_over_index(self, mu, coupling_type):
        r"""Compile coupling coefficients with the first triplet index fixed.

        Parameters
        ----------
        mu : tuple or list [of length 3] of int
            Fixed triplet index.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.

        Returns
        -------
        couplings_vector : complex or float, array_like
            Vector of coupling coefficients with first triplet index fixed.

        Raises
        ------
        ValueError
            If `coupling_type` does not correspond to a valid kernel.

        Notes
        -----
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

        """
        if coupling_type.lower().startswith('a'):
            _sigma_gen = lambda deg_idx, ell: [
                (ell, m, None) for m in range(-ell, ell+1)
            ]
        elif coupling_type.lower().startswith(('rad', 'rsd')):
            _sigma_gen = lambda deg_idx, ell: [
                (ell, None, n) for n in range(1, self.disc.depths[deg_idx]+1)
            ]
        else:
            raise ValueError(
                "`coupling_type` can only be: 'angular', 'radial' or 'RSD'. "
            )

        couplings_vector = []
        for ell_idx, ell in enumerate(self.disc.degrees):
            sigma_ell = _sigma_gen(ell_idx, ell)
            sigma_component = [
                self.__call__(mu, sigma, 'angular') for sigma in sigma_ell
            ]
            couplings_vector.append(np.array(sigma_component))

        return couplings_vector


# 2-Point Correlators
# -----------------------------------------------------------------------------

# TODO: Generalise for all indices as a class derived from ``disc``.
def two_point_signal(mu_couplings, nu_couplings, power_spectrum, beta0, disc):
    r"""Compute the 2-point function signal from linear power spectrum model
    for given indices specified by the coupling vectors.

    Parameters
    ----------
    mu_couplings, nu_couplings : dict
        Compiled coupling coefficients for each of the coupling types for
        ``mu`` or ``nu``, the implicitly specified indices.
    power_spectrum : callable
        Linear galaxy-clustering power spectrum model (length unit Mpc/h).
    beta0 : float
        Linear growth rate over bias :math:`\beta_0` at the current epoch.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum for which the 2-point function is evaluated.

    Returns
    -------
    signal : complex
        Cosmological signal 2-point function value for given triple indices.

    Notes
    -----

    The input coupling vectors have a nested-list structure may be called from:

    ::

        mu_couplings = {
            coupling_type: Couplings.compile_over_index(mu, coupling_type)
            for coupling_type in ['angular', 'radial', 'RSD']
        }

    See :class:`Couplings` for details.  If :class:`Couplings` is instantiated
    with a nontrivial `cosmo_specs`, the cosmology associated with
    `cosmo_specs` must agree with that associated with `power_spectrum` and
    `beta0`.  Similarly, `mu_couplings` and `nu_couplings` must be compatible
    with the discretisation `disc`.

    """
    k, kappa = disc.wavenumbers, disc.normalisation

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
                (Phi_mu_[n_idx] + beta0 * Upsilon_mu_[n_idx])
                    * (Phi_nu_[n_idx] + beta0 * Upsilon_nu_[n_idx])
                    * power_spectrum(k_ell[n_idx]) / kappa_ell[n_idx]
                for n_idx in range(disc.depths[ell_idx])
            ]
        )
        signal += angular_sum * radial_sum

    return signal


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
        Selection or weight as a function of the radial coordinate (default is
        `None`).

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

    if not hasattr(selection, '__call__') and not hasattr(weight, '__call__') \
            and ell_mu == ell_nu:
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
