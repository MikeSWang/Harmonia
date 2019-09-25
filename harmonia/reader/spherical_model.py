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
    `k_elln` and `normalisation`, are assumed to be in the natural structure
    starting at spherical degree :math:`\ell = 0`, so `ellidx` is equal to
    `ell` (see :class:`~harmonia.algorithms.morph.SphericalArray`).  In the
    future, these variables may be changed to :obj:`dict` without assuming this
    correspondence, and relevant :math:`\ell`-modes are accessed through keys.

Kernels
-------------------------------------------------------------------------------

Coupling kernels are integrands without the coordinate Jacobian, which may
include: radial selection :math:`\phi(r)`, weight :math:`w(r)`, and(weighted)
angular mask :math:`M(\hat{\mathbf{r}})`; linear growth rate normalised to
linear bias :math:`\beta(z) = f(z)/b(z)`, clustering evolution
:math:`G(z) = b(z) D(z)` where :math:`D(z)` is the linear growth factor,
and the Alcock--Paczynski distortion

.. math::

    \gamma(z) = \frac{\beta(z)}{\beta_0}
    \frac{\mathrm{d}\tilde{r}}{\mathrm{d}r} \,, \quad
    \text{with} \quad \beta_0 \equiv \beta(0) \,.

When using integration kernels that is a combination of functions such as
weight, selection, mask and evolution etc., pass additional parameters not
being integrated over by redefining these functions with
:func:`functools.partial` or :obj:`lambda`.

.. autosummary::

    angular_kernel
    radial_kernel
    RSD_kernel
    shot_noise_kernel

Couplings
-------------------------------------------------------------------------------

Coupling coefficients are computed by integrating, over spherical coordinates
and Jacobians, the angular, radial and RSD coupling kernels

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

where :math:`\tilde{r}` is the distance converted in a fiducial cosmological
model rather than from the true comoving distance--redshift correspondence, and
:math:`\{ k_{\ell n} \}` are the discrete wavenumbers.

Numerical integration is performed with
:mod:`~harmonia.algorithms.integration`.

.. autosummary::

    angular_coupling
    radial_coupling
    RSD_coupling
    coupling_list

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

"""
import warnings
from functools import partial

import numpy as np

from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic
from harmonia.algorithms.integration import (
    angular_spherical_integral as ang_int,
    radial_spherical_integral as rad_int,
)


# KERNELS
# -----------------------------------------------------------------------------

def angular_kernel(theta, phi, mu, nu, mask=None):
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


def radial_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
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


def RSD_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
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

    if AP_distortion is not None:
        if not hasattr(r2z, '__call__'):
            raise ValueError("`r2z` must be callable if `AP_distortion` is. ")
        kernel *= AP_distortion(r2z(r))

    return kernel


def shot_noise_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None):
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

def angular_coupling(mu, nu, mask=None):
    r"""Compute angular coupling coefficients :math:`M_{\mu\nu}`.

    When there is no angular masking (i.e. `mask` is `None`), the coupling
    coefficients reduce to :math:`M_{\mu\nu} = \delta_{\mu\nu}`.

    Parameters
    ----------
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    mask : callable or None, optional
        `mask` as a keyword argument to be passed to :func:`angular_kernel`
        (default is `None`).

    Returns
    -------
    complex, array_like
        Angular coupling coefficients for given indices.

    """
    if mask is None:
        if mu[0] == nu[0] and mu[1] == nu[1]:
            return 1. + 0.j
        return 0. + 0.j

    return ang_int(
        lambda theta, phi: angular_kernel(theta, phi, mu, nu, mask=mask),
    )


def radial_coupling(mu, nu, rmax, k_elln, normalisation, selection=None,
                    weight=None, evolution=None, r2z=None, z2chi=None):
    r"""Compute angular coupling coefficients :math:`\Phi_{\mu\nu}`.

    When there is no angular masking or clustering evolution, if radial
    selection and weighting are also both absent and the distance--redshift
    conversion is the true one (i.e. `mask`, `selection`, `weight`, `evolution`, `r2z`
    and `z2chi` are all `None`), the coupling coefficients reduce to
    :math:`\Phi_{\mu\nu} = \delta_{\mu\nu}`.

    Parameters
    ----------
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    rmax : float
        Radial integration upper limit.
    k_elln : float, array_like
        Discrete wave numbers.
    normalisation : float, array_like
        Normalisation coefficients.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default is
        `None`).
    evolution : callable or None, optional
        Evolution function of redshift (default is `None`).
    r2z : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default is
        `None`).
    z2chi : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is `None`).

    Returns
    -------
    float, array_like
        Radial coupling coefficients for given indices.

    """
    if (all(func is None for func in [selection, weight, evolution, r2z, z2chi])
            and mu[0] == nu[0]):
        if mu[-1] == nu[-1]:
            return 1.
        return 0.

    k_mu, k_nu = k_elln[mu[0]][mu[-1]-1], k_elln[nu[0]][nu[-1]-1]
    kappa_nu = normalisation[nu[0]][nu[-1]-1]
    funcs = dict(
        selection=selection, weight=weight, evolution=evolution, r2z=r2z,
        z2chi=z2chi,
        )

    kernel = partial(
        radial_kernel, mu=mu, nu=nu, k_mu=k_mu, k_nu=k_nu, **funcs
        )

    return kappa_nu * rad_int(kernel, rmax)


def RSD_coupling(mu, nu, rmax, k_elln, normalisation, selection=None, weight=None,
                 weight_derivative=None, evolution=None, AP_distortion=None, r2z=None, z2chi=None):
    r"""Compute RSD coupling coefficients :math:`\Upsilon_{\mu\nu}`.

    Parameters
    ----------
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    rmax : float
        Radial integration upper limit.
    k_elln : float, array_like
        Discrete wave numbers.
    normalisation : float, array_like
        Normalisation coefficients.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default is
        `None`).
    weight_derivative : callable or None, optional
        Weight function derivative as a function of the radial coordinate
        (default is `None`).
    evolution, AP_distortion : callable or None, optional
        Evolution or AP distortion as a function of redshift (default is
        `None`).
    r2z, z2chi : callable or None, optional
        Conversion function from comoving distance to redshift or vice versa
        (default is `None`).

    Returns
    -------
    float, array_like
        RSD coupling coefficients for given indices.

    """
    k_mu, k_nu = k_elln[mu[0]][mu[-1]-1], k_elln[nu[0]][nu[-1]-1]
    kappa_nu = normalisation[nu[0]][nu[-1]-1]
    funcs = dict(
        selection=selection, weight=weight, weight_derivative=weight_derivative, evolution=evolution,
        AP_distortion=AP_distortion, r2z=r2z, z2chi=z2chi,
        )

    kernel = partial(RSD_kernel, mu=mu, nu=nu, k_mu=k_mu, k_nu=k_nu, **funcs)

    return kappa_nu / k_nu * rad_int(kernel, rmax)


def coupling_list(mu, coupletype, disc, **funcs):
    r"""Compile list of coupling coefficients with given fixed indices.

    For coupling coefficients of the form :math:`C_{a_\mu a_\sigma b_\mu
    b_\sigma}`, this function takes fixed :math:`\mu` and computes their values
    through for all tuples indexed by :math:`\sigma`.  For example,

    ::

        coupling_list(mu, 'ang', disc, **funcs)

    returns the quantity

    .. math::

        M_{\ell_\mu m_\mu \cdot \cdot}
        = {\left\lbrace
               M_{\ell_\mu m_\mu \ell_\sigma m_\sigma}
               \,\middle\vert\,
               m_\sigma = -\ell_\sigma, \dots, \ell_\sigma
           \right\rbrace}_{\sigma} \,.

    Parameters
    ----------
    mu : tuple or list [of length 3] of int
        Fixed triplet index.
    coupletype : {'ang', 'rad', 'rsd'}
        Coupling function to be evaluated and compiled, with ``'ang'`` for
        :func:`angular_coupling`, ``'rad'`` for :func:`radial_coupling` and
        ``'rsd'`` for :func:`RSD_coupling`.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    **funcs : callable, optional
        Additional functions as keyword arguments to be passed to coupling
        functions: `mask` to :func:`angular_coupling`; `selection`, `weight`,
        `evolution`, `r2z`, `z2chi`, `mask` to :func:`radial_coupling`, and
        similarly to :func:`RSD_coupling` with additionally `weight_derivative`.  If none
        is passed, default values for these keyword arguments are used in the
        defitions of these coupling functions.

    Returns
    -------
    C_mu_all : complex or float, array_like
        List of coupling coefficients with first triplet index fixed.

    Raises
    ------
    ValueError
        If `coupletype` is not one of the allowed strings.

    """
    case = coupletype.lower()[:3]
    if case not in ['ang', 'rad', 'rsd']:
        raise ValueError("`coupletype` must be one of 'ang', 'rad' or 'rsd'. ")

    rmax = disc.attrs['boundary_radius']
    ells, nmax = disc.degrees, disc.depths
    k_elln, kappa_elln = disc.wavenumbers, disc.normalisation

    C_mu_all = []
    for ellidx, ell_sigma in enumerate(ells):
        if case == 'ang':
            C_musigma = np.array(
                [angular_coupling(mu, (ell_sigma, m_sigma, None), **funcs)
                 for m_sigma in range(-ell_sigma, ell_sigma+1)]
                )
        elif case in ['rad', 'rsd']:
            if case == 'rad':
                coupling = radial_coupling
            elif case == 'rsd':
                coupling = RSD_coupling
            C_musigma = np.array(
                [coupling(
                    mu, (ell_sigma, None, n_sigma), rmax, k_elln, kappa_elln,
                    **funcs
                    )
                 for n_sigma in range(1, nmax[ellidx]+1)
                 ]
                )
        C_mu_all.append(C_musigma)

    return C_mu_all


# 2-Point Correlators
# -----------------------------------------------------------------------------

# TODO: Generalise for all indices as a class derived from ``disc``.
def two_point_signal(pklin, beta0, disc, M_mu_all=None, M_nu_all=None,
                    Phi_mu_all=None, Phi_nu_all=None, Upsilon_mu_all=None,
                    Upsilon_nu_all=None):
    r"""Compute the 2-point function signal from linear power spectrum model.

    Parameters
    ----------
    pklin : callable
        Linear galaxy-clustering power spectrum model (length unit Mpc/h).
    beta0 : float
        Linear growth rate over bias :math:`\beta_0` at the current epoch.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    M_mu_all, M_nu_all : nested list of complex, array_like, optional
        Angular coupling coefficients with one triple index being `mu` or
        `nu` (default is `None`).
    Phi_mu_all, Phi_nu_all: nested list of float, optional
        Raidal coupling coefficients with one triple index being `mu` or
        `nu` (default is `None`).
    Upsilon_mu_all, Upsilon_nu_all: nested list of float, optional
        RSD coupling coefficients with one triple index being `mu` or
        `nu` (default is `None`).

    Returns
    -------
    signal2pt_munu: complex
        Cosmological signal 2-point function value for given triple indices.

    Notes
    -----
    The input coupling coefficients have a particular structure, and a list of
    arrays of such may be obtained from:

    ::

        M_mu_all = coupling_list(mu, 'ang', disc, **funcs)

    Warnings
    --------
    `pklin` and `beta0` must have the same underlying cosmology.

    """
    ells, nmaxs = disc.degrees, disc.depths
    k_elln, kappa_elln = disc.wavenumbers, disc.normalisation

    # Perform summation.
    signal2pt_munu = 0
    for ell_sigma in ells:
        # Angular summation.
        ang_sum = np.sum(
            [M_mu_all[ell_sigma][midx] * np.conj(M_nu_all[ell_sigma][midx])
             for midx in range(0, 2*ell_sigma+1)
             ]
            )

        # Radial summation.
        rad_sum = np.sum(
            [(Phi_mu_all[ell_sigma][nidx]
              + beta0 * Upsilon_mu_all[ell_sigma][nidx]) \
             * (Phi_nu_all[ell_sigma][nidx]
                + beta0 * Upsilon_nu_all[ell_sigma][nidx]) \
             * pklin(k_elln[ell_sigma][nidx]) / kappa_elln[ell_sigma][nidx]
             for nidx in range(nmaxs[ell_sigma])
             ]
            )

        signal2pt_munu += ang_sum * rad_sum

    return signal2pt_munu


def two_point_shot_noise(mu, nu, nmean, disc, M_munu, selection=None, weight=None):
    r"""Compute the shot noise 2-point function.

    Parameters
    ----------
    mu, nu : tuple or list [of length 3] of int
        Coefficient triplet index.
    nmean : float
        Sampled homogeneous mean particle number density (length unit Mpc/h).
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    M_munu : complex
        Angular mask coupling coefficients :math:`M_{\mu\nu}`.
    selection, weight : callable or None, optional
        Selection or weight as a function of the radial coordinate (default is
        `None`).

    Returns
    -------
    complex, array_like
        Shot noise 2-point function value for given indices.

    """
    if np.allclose(M_munu, 0.):
        return 0.

    ellidx_mu, ellidx_nu, nidx_mu, nidx_nu = mu[0], nu[0], mu[-1]-1, nu[-1]-1

    u_elln, k_elln = disc.wavenumbers, disc.roots
    rmax = disc.attrs['boundary_radius']

    u_mu = u_elln[ellidx_mu][nidx_mu]
    k_mu, k_nu = k_elln[ellidx_mu][nidx_mu], k_elln[ellidx_nu][nidx_nu]

    if (selection is None) and (weight is None) and (mu[0] == nu[0]):
        if mu[-1] == nu[-1]:
            return M_munu/nmean * rmax**3/2 * spherical_besselj(ellidx_mu+1, u_mu)**2
        return 0.

    kernel = partial(
        shot_noise_kernel, mu=mu, nu=nu, k_mu=k_mu, k_nu=k_nu, selection=selection,
        weight=weight
        )

    return M_munu/nmean * rad_int(kernel, rmax)
