r"""
Spherical Fourier-space model (:mod:`~harmonia.reader.spherical_model`)
===============================================================================

Compute the spherical model for Fourier-space :math:`n`-point functions.  The
index for each \"point\" is a triplet labelled by a Greek letter, e.g.
:math:`(\ell_\mu, m_\mu, n_\mu)` as ``(mu[0], mu[1], mu[2])``.

.. _ellidx-warning:

.. topic:: Important notice

    All variables related to the discretised spectrum
    :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`, such as
    `k_elln` and `normcoeff`, are assumed be in the natural structure (see
    :class:`~harmonia.algorithms.morph.SphericalArray`) starting at spherical
    degree :math:`\ell = 0` in this module, so `ellidx` is equal to `ell`.  In
    future, these variables may be changed to :obj:`dict` without assuming this
    correspondence, and relevant :math:`\ell`-modes are accessed through keys.

Kernels
-------------------------------------------------------------------------------

Coupling kernels are integrands without the spherical integration Jacobian,
which may include radial selection and weight functions :math:`\phi(r), w(r)`,
and (weighted) angular mask :math:`M(\hat{\mathbf{r}})` of the spatial
position; and as functions of redshift the linear growth rate normalised to
linear bias :math:`\beta(z) = f(z)/b(z)`, clustering evolution function
:math:`G(z) = b(z) D(z)` where :math:`D(z)` is the linear growth factor and the
*Alcock--Paczynski distortion function*

.. math::

    \gamma(z) = \frac{\beta(z)}{\beta_0}
    \frac{\mathrm{d}\tilde{r}}{\mathrm{d}r} \,,

where :math:`\beta_0 \equiv \beta(0)` at the current epoch :math:`z = 0`.

When using integration kernels that is a combination of functions such as
weight, selection, mask and evolution etc., pass additional parameters not
being integrated over by redefining these functions with
:func:`functools.partial`.  For instance, for a radial selection function `sel`
of integration variable `r` and additional parameters ``*args``, ``**kargs``:

::

    from functools import partial

    sel = partial(sel, *args, **kargs)

.. autosummary::

    angular_kernel
    radial_kernel
    rsd_kernel
    shotnoise_kernel

Couplings
-------------------------------------------------------------------------------

Coupling coefficients are computed by integrating over spherical coordinates
the angular, radial and RSD kernels (modulo Jacobians)

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
:math:`\{k_{\ell n}\}` are the discrete wavenumbers.

Numerical integration is performed with
:mod:`~harmonia.algorithms.integration`.

.. autosummary::

    angular_coupling
    radial_coupling
    rsd_coupling
    coupling_list

2-point correlators
-------------------------------------------------------------------------------

2-point correlators are computed from couplings as a sum of the signal part

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

    twopoint_signal
    twopoint_shotnoise

.. todo:: Under development.

"""
import warnings
from functools import partial

import numpy as np

from harmonia.algorithms.bases import sph_besselj, sph_harmonic
from harmonia.algorithms.integration import (
    angular_spherical_int as ang_sphint, radial_spherical_int as rad_sphint,
    )


# KERNELS
# -----------------------------------------------------------------------------

def angular_kernel(theta, phi, mu, nu, mask_func=None):
    r"""Compute the angular coupling kernel.

    Parameters
    ----------
    theta, phi : float, array_like
        Spherical angular coordinates :math:`(\theta, \phi)`.
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    mask_func : callable or None, optional
        Mask as a function of spherical angular coordinates (default is
        `None`).

    Returns
    -------
    kernel : complex, array_like
        Angular coupling kernel.

    Warnings
    --------
    The first two positional arguments of `mask_func` must be in radians and in
    the following order and range: :math:`(\theta, \phi) \in [0, \pi] \times
    [0, 2\pi)`.

    """
    kernel = np.conj(sph_harmonic(mu[0], mu[1], theta, phi)) \
        * sph_harmonic(nu[0], nu[1], theta, phi)

    if mask_func is not None:
        kernel *= mask_func(theta, phi)
    else:
        warnings.warn(
            "`mask_func` is `None`. "
            "Angular model evaluation may be redundant. ",
            RuntimeWarning
            )

    return kernel


def radial_kernel(r, mu, nu, k_mu, k_nu, sel_func=None, wgt_func=None,
                  evo_func=None, dist2z_func=None, z2chi_func=None):
    r"""Compute the radial coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Spherical radial coordinate.
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    k_mu, k_nu : float
        Discretised spherical wave numbers corresponding to indices `mu`, `nu`.
    sel_func, wgt_func : callable or None, optional
        Weight or selection as a function of the spherical radial coordinate
        (default is `None`).
    evo_func : callable or None, optional
        Evolution function :math:`G(z)` of redshift (default is `None`).
    dist2z_func : callable or None, optional
        'True' cosmological comoving distance-to-redshift conversion (default
        is `None`).
    z2chi_func : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is `None`).

    Returns
    -------
    kernel : float, array_like
        Radial coupling kernel.

    Raises
    ------
    ValueError
        If `dist2z_func` is `None` when `evo_func` or `z2chi_func` is not
        `None`.

    """
    r_tilde = r
    if z2chi_func is not None:
        if dist2z_func is not None:
            r_tilde = z2chi_func(dist2z_func(r))
        else:
            raise ValueError(
                "`dist2z_func` cannot be `None` "
                "if `z2chi_func` is not `None`. "
                )

    kernel = sph_besselj(mu[0], k_mu*r_tilde) * sph_besselj(nu[0], k_nu*r)
    if sel_func is not None:
        kernel *= sel_func(r)
    if wgt_func is not None:
        kernel *= wgt_func(r_tilde)
    if evo_func is not None:
        if dist2z_func is None:
            raise ValueError(
                "`dist2z_func` cannot be `None` if `evo_func` is not `None`. "
                )
        kernel *= evo_func(dist2z_func(r))

    return kernel


def rsd_kernel(r, mu, nu, k_mu, k_nu, sel_func=None, wgt_func=None,
               wgt_deriv=None, evo_func=None, gamma_func=None,
               dist2z_func=None, z2chi_func=None):
    r"""Compute the RSD coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Spherical radial coordinate.
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    k_mu, k_nu : float
        Discretised spherical wave numbers corresponding to indices `mu`, `nu`.
    sel_func, wgt_func : callable or None, optional
        Weight or selection as a function of the spherical radial coordinate
        (default is `None`).
    wgt_deriv : callable or None, optional
        Weight function derivative as a function of the spherical radial
        coordinate (default is `None`).
    evo_func, gamma_func : callable or None, optional
        Evolution and AP fiducial conversion distortion functions of redshift
        (default is `None`).
    dist2z_func : callable or None, optional
        'True' cosmological comoving distance-to-redshift conversion (default
        is `None`).
    z2chi_func : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is `None`).

    Returns
    -------
    kernel : float, array_like
        RSD coupling kernel.

    Raises
    ------
    ValueError
        If `wgt_deriv` is `None` when `wgt_func` is not `None`.
    ValueError
        If `dist2z_func` is `None` when `z2chi_func`, `evo_func` or
        `gamma_func` is not `None`.

    """
    r_tilde = r
    if z2chi_func is not None:
        if dist2z_func is not None:
            r_tilde = z2chi_func(dist2z_func(r))
        else:
            raise ValueError(
                "`dist2z_func` cannot be `None` "
                "if `z2chi_func` is not `None`. "
                )

    kernel = sph_besselj(nu[0], k_nu*r, deriv=True)
    if sel_func is not None:
        kernel *= sel_func(r)
    if wgt_func is None:
        kernel *= k_mu * sph_besselj(mu[0], k_mu*r, deriv=True)
    else:
        if wgt_deriv is None:
            raise ValueError(
                "`wgt_deriv` cannot be `None` if `wgt_func` is not `None`. "
                )
        kernel *= (
            wgt_deriv(r_tilde) * sph_besselj(mu[0], k_mu*r_tilde)
            + k_mu * wgt_func(r_tilde) * sph_besselj(
                mu[0], k_mu*r_tilde, deriv=True
                )
            )
    if evo_func is not None:
        if dist2z_func is None:
            raise ValueError(
                "`dist2z_func` cannot be `None` if `evo_func` is not `None`. "
                )
        kernel *= evo_func(dist2z_func(r))
    if gamma_func is not None:
        if dist2z_func is None:
            raise ValueError(
                "`dist2z_func` cannot be `None` "
                "if `gamma_func` is not `None`. "
                )
        kernel *= gamma_func(dist2z_func(r))

    return kernel


def shotnoise_kernel(r, mu, nu, k_mu, k_nu, sel_func=None, wgt_func=None):
    r"""Compute the integral kernel for the shot noise 2-point correlator.

    Parameters
    ----------
    r : float, array_like
        Spherical radial coordinate.
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    k_mu, k_nu : float
        Discretised spherical wave numbers corresponding to indices `mu`, `nu`.
    sel_func, wgt_func : callable or None, optional
        Weight or selection as a function of the spherical radial coordinate
        (default is `None`).

    Returns
    -------
    kernel : float, array_like
        Shot noise 2-point function kernel.

    """
    if (sel_func is None) and (wgt_func is None) and (mu[0] == nu[0]):
        warnings.warn(
            "Shot noise evaluation may be redundant. ", RuntimeWarning
            )

    kernel = sph_besselj(mu[0], k_mu*r) * sph_besselj(nu[0], k_nu*r)
    if sel_func is not None:
        kernel *= sel_func(r)
    if wgt_func is not None:
        kernel *= wgt_func(r)**2

    return kernel


# COUPLINGS
# -----------------------------------------------------------------------------

def angular_coupling(mu, nu, mask=None):
    r"""Compute angular coupling coefficients :math:`M_{\mu\nu}`.

    When there is no angular masking (i.e. `mask` is `None`), the coupling
    coefficients reduce to :math:`M_{\mu\nu} = \delta_{\mu\nu}`.

    Parameters
    ----------
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    mask : callable or None, optional
        `mask` as a keyword argument to be passed to :func:`angular_kernel`
        (default is `None`).

    Returns
    -------
    complex, array_like
        Angular coupling coefficients.

    """
    if mask is None:
        if mu[0] == nu[0] and mu[1] == nu[1]:
            return 1. + 0.j
        return 0. + 0.j

    kernel = partial(angular_kernel, mu=mu, nu=nu, mask_func=mask)

    return ang_sphint(kernel)


def radial_coupling(mu, nu, rmax, k_elln, normcoeff, sel=None,
                    wgt=None, evo=None, dist2z=None, z2chi=None):
    r"""Compute angular coupling coefficients :math:`\Phi_{\mu\nu}`.

    When there is no angular masking or clustering evolution, if radial
    selection and weighting are also both absent and the distance--redshift
    conversion is the true one (i.e. `mask_func`, `sel`, `wgt`, `evo`, `dist2z`
    and `z2chi` are all `None`), the coupling coefficients reduce to
    :math:`\Phi_{\mu\nu} = \delta_{\mu\nu}`.

    Parameters
    ----------
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    rmax : float
        Radial integration upper limit.
    k_elln : float, array_like
        Discrete wave numbers.
    normcoeff : float, array_like
        Normalisation coefficients.
    sel, wgt : callable or None, optional
        Weight or selection as a function of the spherical radial coordinate
        (default is `None`).
    evo : callable or None, optional
        Evolution function of redshift (default is `None`).
    dist2z : callable or None, optional
        'True' cosmological comoving distance-to-redshift conversion (default
        is `None`).
    z2chi : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is `None`).

    Returns
    -------
    float, array_like
        Radial coupling coefficients.

    """
    if (all(func is None for func in [sel, wgt, evo, dist2z, z2chi])
            and mu[0] == nu[0]):
        if mu[-1] == nu[-1]:
            return 1.
        return 0.

    k_mu, k_nu = k_elln[mu[0]][mu[-1]-1], k_elln[nu[0]][nu[-1]-1]
    kappa_nu = normcoeff[nu[0]][nu[-1]-1]
    funcs = dict(
        sel_func=sel, wgt_func=wgt, evo_func=evo, dist2z_func=dist2z,
        z2chi_func=z2chi,
        )

    kernel = partial(
        radial_kernel, mu=mu, nu=nu, k_mu=k_mu, k_nu=k_nu, **funcs
        )

    return kappa_nu * rad_sphint(kernel, rmax)


def rsd_coupling(mu, nu, rmax, k_elln, normcoeff, sel=None, wgt=None,
                 dwgt=None, evo=None, gamma=None, dist2z=None, z2chi=None):
    r"""Compute RSD coupling coefficients :math:`\Upsilon_{\mu\nu}`.

    Parameters
    ----------
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    rmax : float
        Radial integration upper limit.
    k_elln : float, array_like
        Discrete wave numbers.
    normcoeff : float, array_like
        Normalisation coefficients.
    sel, wgt : callable or None, optional
        Weight or selection as a function of the spherical radial coordinate
        (default is `None`).
    dwgt : callable or None, optional
        Weight function derivative as a function of the spherical radial
        coordinate (default is `None`).
    evo, gamma : callable or None, optional
        Evolution and AP fiducial conversion distortion functions of redshift
        (default is `None`).
    dist2z, z2chi : callable or None, optional
        Conversion function from comoving distance to redshift or vice versa
        (default is `None`).

    Returns
    -------
    float, array_like
        RSD coupling coefficients.

    """
    k_mu, k_nu = k_elln[mu[0]][mu[-1]-1], k_elln[nu[0]][nu[-1]-1]
    kappa_nu = normcoeff[nu[0]][nu[-1]-1]
    funcs = dict(
        sel_func=sel, wgt_func=wgt, wgt_deriv=dwgt, evo_func=evo,
        gamma_func=gamma, dist2z_func=dist2z, z2chi_func=z2chi,
        )

    kernel = partial(rsd_kernel, mu=mu, nu=nu, k_mu=k_mu, k_nu=k_nu, **funcs)

    return kappa_nu / k_nu * rad_sphint(kernel, rmax)


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
    mu : tuple or list (of length 3) of int
        Fixed triplet index.
    coupletype : {'ang', 'rad', 'rsd'}
        Coupling function to be evaluated and compiled, with ``'ang'`` for
        :func:`angular_coupling`, ``'rad'`` for :func:`radial_coupling` and
        ``'rsd'`` for :func:`rsd_coupling`.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    **funcs : callable, optional
        Additional functions as keyword arguments to be passed to coupling
        functions: `mask_func` to :func:`angular_coupling`; `sel`, `wgt`,
        `evo`, `dist2z`, `z2chi`, `mask` to :func:`radial_coupling`, and
        similarly to :func:`rsd_coupling` with additionally `dwgt`.  If none
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
    k_elln, kappa_elln = disc.wavenumbers, disc.normcoeff

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
                coupling = rsd_coupling
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
def twopoint_signal(pklin, beta0, disc, M_mu_all=None, M_nu_all=None,
                    Phi_mu_all=None, Phi_nu_all=None, Upsilon_mu_all=None,
                    Upsilon_nu_all=None):
    r"""Compute the 2-pont correlator signal from linear power spectrum model.

    Parameters
    ----------
    pklin : callable
        Linear galaxy-clustering power spectrum model.
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
        Cosmological signal 2-point correlator value for given triple indices.

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
    k_elln, kappa_elln = disc.wavenumbers, disc.normcoeff

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


def twopoint_shotnoise(mu, nu, nmean, disc, M_munu, sel=None, wgt=None):
    r"""Compute the shot noise 2-pont correlator.

    Parameters
    ----------
    mu, nu : tuple or list (of length 3) of int
        Coefficient triplet indices.
    nmean : float
        Sampled homogeneous mean particle number density.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    M_munu : complex
        Angular mask coupling coefficients :math:`M_{\mu\nu}`.
    sel, wgt : callable or None, optional
        Weight or selection as a function of the spherical radial coordinate
        (default is `None`).

    Returns
    -------
    complex, array_like
        Shot noise 2-point correlator value for given triple indices.

    """
    if np.allclose(M_munu, 0.):
        return 0.

    ellidx_mu, ellidx_nu, nidx_mu, nidx_nu = mu[0], nu[0], mu[-1]-1, nu[-1]-1

    u_elln, k_elln = disc.wavenumbers, disc.roots
    rmax = disc.attrs['boundary_radius']

    u_mu = u_elln[ellidx_mu][nidx_mu]
    k_mu, k_nu = k_elln[ellidx_mu][nidx_mu], k_elln[ellidx_nu][nidx_nu]

    if (sel is None) and (wgt is None) and (mu[0] == nu[0]):
        if mu[-1] == nu[-1]:
            return M_munu/nmean * rmax**3/2 * sph_besselj(ellidx_mu+1, u_mu)**2
        return 0.

    kernel = partial(
        shotnoise_kernel, mu=mu, nu=nu, k_mu=k_mu, k_nu=k_nu, sel_func=sel,
        wgt_func=wgt
        )

    return M_munu/nmean * rad_sphint(kernel, rmax)
