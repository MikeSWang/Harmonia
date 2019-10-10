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
:math:`M(\hat{\mathbf{r}})`; linear growth rate normalised to constant
linear bias :math:`\beta(z) = f(z)/b_1(z)`, clustering evolution
:math:`G(z) = b_1(z) D(z)` where :math:`D(z)` is the linear growth factor,
and the Alcock--Paczynski distortion

.. math::

    \gamma(z) = \frac{\mathrm{d}\tilde{r}(z)}{\mathrm{d}r(z)} \,,

where :math:`\tilde{r} = \tilde{r}(z)` is the fiducial distance converted
from redshift rather than from the true comoving distance--redshift
correspondence :math:`z = z(r)`.

When using integration kernels that is a combination of functions of the
above, pass additional parameters not being directly integrated over by
redefining these functions with the `lambda` keyword.

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
   j'_{\ell_\nu}(k_{\ell_\nu n_\nu} r) \gamma(z) f(z) G(z) \phi(r) \,,

over the spherical volume element, where :math:`k_{\ell n}` are the
discrete wavenumbers.

.. autosummary::

    Couplings

2-point functions
---------------------------------------------------------------------------

2-point functions are computed from couplings as a sum of the signal part

.. math::

    \left\langle \delta_\mu \delta_\nu \right\rangle = \sum_\sigma
        M_{\mu\sigma} M^*_{\nu\sigma} \left[
            b_0(k_\sigma) \Phi_{\mu\sigma} + f_0 \Upsilon_{\mu\sigma}
        \right] \left[
            b_0(k_\sigma) \Phi_{\nu\sigma} + f_0 \Upsilon_{\nu\sigma}
        \right] \kappa_\sigma^{-1} P(k_\sigma) \,,

where :math:`b_0(k)` is the scale-dependent modification of the constant
linear bias :math:`b_1(z=0)` at the current epoch, and the shot noise part

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
from nbodykit.lab import cosmology

from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic
from harmonia.algorithms.integration import (
    angular_spherical_integral as ang_int,
    radial_spherical_integral as rad_int,
)
from harmonia.algorithms.morph import SphericalArray
from harmonia.collections.utils import const_function, mpi_compute
from harmonia.cosmology.scale_dependence import scale_dependent_bias


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

    if callable(mask):
        kernel *= mask(theta, phi)
    else:
        warnings.warn(
            "`mask` is None. Angular model evaluation may be redundant. ",
            RuntimeWarning
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
            RuntimeWarning
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
    survey_specs : dict of {str: callable or None} or None, optional
        Survey specification functions accessed with the following
        mandatory keys: ``'mask'`` for angular mask, and ``'selection'``
        and ``'weight'`` for radial selection and weighting,
        ``'weight_derivative'`` for the derivative function of radial
        weighting.  Default is `None`.
    cosmo_specs : dict of {str: callable or None} or None, optional
        Cosmological specification functions accessed with the following
        mandatory keys: ``'r2z'`` for cosmological comoving
        distance-to-redshift conversion, ``'z2chi'`` for fiducial
        redshift-to-comoving distance conversion, ``'evolution'`` for
        clustering evolution, and ``'AP_distortion'`` for AP distortion.
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
    AP_distortion : callable or None
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
        "survey_specs": ('mask', 'selection', 'weight', 'weight_derivative'),
        "cosmo_specs": ('r2z', 'z2chi', 'evolution', 'AP_distortion'),
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
                        if attr_val is not None and not callable(attr_val):
                            raise TypeError(
                                f"{specs_var_str} {attr} value"
                                f" must be None or callable. "
                            )
                except KeyError as missing_key:
                    raise KeyError(
                        f"{specs_var_str} key {missing_key} is missing. "
                    )
            elif specs is None:
                for attr in specs_attrs:
                    setattr(self, attr, None)


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
        mu, nu : tuple or list of int
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
        self._logger.debug("Computing %s. ", _info_msg)

        if coupling_type.lower().startswith('a'):
            trivial_case = not callable(self.mask)
            if trivial_case:
                if mu[0] == nu[0] and mu[1] == nu[1]:
                    return 1. + 0.j
                return 0. + 0.j

            def _ang_kernel(theta, phi):

                return _angular_kernel(theta, phi, mu, nu, mask=self.mask)

            coupling_coeff = ang_int(_ang_kernel)
            self._logger.debug("Computed %s. ", _info_msg)

            return coupling_coeff

        rmax = self.disc.attrs['boundary_radius']

        ell_mu, n_mu = mu[0], mu[-1]
        ell_nu, n_nu = nu[0], nu[-1]

        k_mu = self.disc.wavenumbers[ell_mu][n_mu-1]
        k_nu = self.disc.wavenumbers[ell_nu][n_nu-1]
        kappa_nu = self.disc.normalisations[ell_nu][n_nu-1]

        if coupling_type.lower().startswith('rad'):
            attrs = ['selection', 'weight', 'evolution', 'r2z', 'z2chi']
            funcs = {attr: getattr(self, attr) for attr in attrs}

            trivial_case = not any(
                [callable(func) for attr, func in funcs.items()]
            )
            if trivial_case:
                if mu[0] == nu[0]:
                    return float(mu[-1] == nu[-1])  # Kronecker delta

            coupling_coeff = kappa_nu * rad_int(
                lambda r: _radial_kernel(r, mu, nu, k_mu, k_nu, **funcs),
                rmax
            )
            self._logger.debug("Computed %s. ", _info_msg)

            return coupling_coeff

        if coupling_type.lower().startswith('rsd'):
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
                rmax
            )
            self._logger.debug("Computed %s.", _info_msg)

            return coupling_coeff

        raise ValueError(
            "`coupling_type` can only be: 'angular', 'radial' or 'RSD'. "
        )

    def couplings_fixed_index(self, mu, coupling_type):
        r"""Compute coupling coefficients with the first triplet index
        fixed.

        Notes
        -----
        This function computes coupling coefficients of the form
        :math:`C_{a_\mu b_\mu a_\sigma b_\sigma}` where the triplet index
        :math:`\mu` is fixed, and compiles their values as a vector by
        iterating through the triplet index :math:`\sigma`.  For example,

        ::

            Couplings.couplings_fixed_index(mu, 'angular')

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
        couplings_component : dict of {int: :class:`numpy.ndarray`}
            Coupling coefficients with first triplet index fixed as a
            dictionary with integer keys corresponding to spherical
            degrees.

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

        couplings_component = {}
        for ell in self.disc.degrees:
            couplings_component[ell] = np.array(
                [
                    self.coupling_coefficient(mu, sigma, coupling_type)
                    for sigma in sigma_gen(ell)
                ]
            )

        return couplings_component

    def couplings(self, coupling_type, comm=None):
        r"""Compile all coupling coefficients of a given type as a sequence
        iterated through the first triplet index ordered as specified.

        This returns a dictionary whose keys are all the triplet indices,
        each with a value corresponding to the coefficients returned by
        a call of :meth:`~.spherical_model.Couplings.couplings_fixed_index`
        for the specified coupling type and that triplet index.

        Parameters
        ----------
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.
        comm : :class:`mpi4py.MPI.Comm` or None, optional
            MPI communicator.  If `None` (default), no multiprocessing
            is invoked.

        Returns
        -------
        sequenced_couplings : dict of {tuple: |NestedDict|}
            Sequence of coupling coefficients over the first triplet index.


        .. |NestedDict| replace::

            dict of {int: :class:`numpy.ndarray`}

        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index_vector = SphericalArray.build(disc=self.disc)\
                .unfold('natural', return_only='index')

        if comm is not None:
            coeff_processor = lambda mu: \
                self.couplings_fixed_index(mu, coupling_type=coupling_type)
            coeff_vector = mpi_compute(index_vector, coeff_processor, comm)

            if comm.rank == 0:
                sequenced_couplings = dict(zip(index_vector, coeff_vector))
                sequenced_couplings = comm.bcast(sequenced_couplings, root=0)
        else:
            coeff_vector = [
                self.couplings_fixed_index(mu, coupling_type=coupling_type)
                for mu in index_vector
            ]
            sequenced_couplings = dict(zip(index_vector, coeff_vector))

        return sequenced_couplings


# 2-Point Correlators
# -----------------------------------------------------------------------------

class TwoPointFunction(Couplings):
    r"""Compute 2-point function values for given survey and cosmological
    specifications from a biased power spectrum model, linear growth rate
    and local primordial non-Gaussianity.

    Parameters
    ----------
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    b_1 : float
        Constant linear bias of the tracer particles at the current epoch.
    cosmo : :class:`nbodykit.cosmology.Cosmology`
        Cosmological model used to produce the power spectrum model and the
        transfer function for calculating scale-dependent bias.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    f_0 : float or None, optional
        Linear growth rate at the current epoch.  If `None` (default), RSD
        calculations are neglected.
    f_nl : float or None, optional
        Local primordial non-Gaussianity.  If `None` (default), this is set
        to zero and ignored.
    survey_specs : dict of {str: callable or None} or None, optional
        Survey specification functions accessed with the following
        mandatory keys: ``'mask'`` for angular mask, and ``'selection'``
        and ``'weight'`` for radial selection and weighting,
        ``'weight_derivative'`` for the derivative function of radial
        weighting.  Default is `None`.
    cosmo_specs : dict of {str: callable or None} or None, optional
        Cosmological specification functions accessed with the following
        mandatory keys: ``'r2z'`` for cosmological comoving
        distance-to-redshift conversion, ``'z2chi'`` for fiducial
        redshift-to-comoving distance conversion, ``'evolution'`` for
        clustering evolution, and ``'AP_distortion'`` for AP distortion.
        Default is `None`.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

    Attributes
    ----------
    mean_density : float
        Mean particle number density (in cubic h/Mpc).
    bias_const : float
        Constant linear bias at the current epoch.
    growth_rate : float or None
        Linear growth rate at the current epoch.
    non_gaussianity : float or None
        Local primordial non-Gaussianity.
    matter_power_spectrum : |LinearPower|
        Linear matter power spectrum model at the current epoch (in cubic
        Mpc/h).
    comm : :class:`mpi4py.MPI.Comm` or None, optional
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.


    .. |LinearPower| replace::

         :class:`nbodykit.cosmology.power.linear.LinearPower`

    """

    _logger = logging.getLogger("TwoPointFunction")
    _REDSHIFT_EPOCH = 0.

    def __init__(self, nbar, b_1, cosmo, disc, f_0=None, f_nl=None,
                 survey_specs=None, cosmo_specs=None, comm=None):

        super().__init__(
            disc,
            survey_specs=survey_specs,
            cosmo_specs=cosmo_specs
        )

        self.mean_density = nbar
        self.bias_const = b_1
        self.growth_rate = f_0
        self.non_gaussianity = f_nl
        self.matter_power_spectrum = cosmology.LinearPower(
            cosmo,
            redshift=self._REDSHIFT_EPOCH,
            transfer='CLASS'
        )

        self.comm = comm

        self._couplings = None
        if self.non_gaussianity is None:
            self._bias_k = const_function(b_1)
        else:
            self._bias_k = scale_dependent_bias(f_nl, b_1, cosmo)

    @property
    def couplings(self):
        """Coupling coefficients of all types for all indices.

        Returns
        -------
        dict of {str; dict}
            Coupling coefficients as a dictionary for coupling types
            each with a sub-dictionary for all triplet indices.

        """

        if self._couplings is not None:
            return self._couplings

        if self.growth_rate is None:
            self._couplings = dict.fromkeys(['angular', 'radial'])
        else:
            self._couplings = dict.fromkeys(['angular', 'radial', 'RSD'])
        for coupling_type in self._couplings:
            self._couplings[coupling_type] = \
                super().couplings(coupling_type, comm=self.comm)
        self._logger.info("Coupling coefficients computed. ")

        return self._couplings

    def two_point_signal(self, mu, nu):
        """Signal 2-point function for given triplet indices.

        Parameters
        ----------
        mu, nu : tuple or list of int
            Coefficient triplet index.

        Returns
        -------
        signal : complex
            Cosmological signal 2-point function value for given triplet
            indices.

        """
        bias = self._bias_k
        f_0 = self.growth_rate
        pk = self.matter_power_spectrum

        k, kappa = self.disc.wavenumbers, self.disc.normalisations

        couplings = self.couplings

        M_mu, M_nu = couplings['angular'][mu], couplings['angular'][nu]
        Phi_mu, Phi_nu = couplings['radial'][mu], couplings['radial'][nu]
        Upsilon_mu, Upsilon_nu = couplings['RSD'][mu], couplings['RSD'][nu]

        signal = 0
        for ell, nmax in zip(self.disc.degrees, self.disc.depths):
            angular_sum = np.sum(
                [
                    M_mu[ell][m_idx] * np.conj(M_nu[ell][m_idx])
                    for m_idx in range(0, 2*ell+1)
                ]
            )
            if self.growth_rate is None:
                radial_sum = np.sum(
                    [
                        Phi_mu[ell][n_idx] * Phi_nu[ell][n_idx]
                        * bias(k[ell][n_idx])**2
                        * pk(k[ell][n_idx]) / kappa[ell][n_idx]
                        for n_idx in range(0, nmax)
                    ]
                )
            else:
                radial_sum = np.sum(
                    [
                        (bias(k[ell][n_idx]) * Phi_mu[ell][n_idx] \
                             + f_0 * Upsilon_mu[ell][n_idx])
                        * (bias(k[ell][n_idx]) * Phi_nu[ell][n_idx] \
                               + f_0 * Upsilon_nu[ell][n_idx])
                        * pk(k[ell][n_idx]) / kappa[ell][n_idx]
                        for n_idx in range(0, nmax)
                    ]
                )
            signal += angular_sum * radial_sum

        return signal

    def two_point_shot_noise(self, mu, nu):
        """Shot noise 2-point function for given triplet indices.

        Parameters
        ----------
        mu, nu : tuple or list of int
            Coefficient triplet index.

        Returns
        -------
        shot_noise : complex
            Shot noise 2-point function value for given triplet indices.

        """
        rmax = self.disc.attrs['boundary_radius']

        ell_mu, m_mu, n_mu = mu
        ell_nu, m_nu, n_nu = nu

        M_mu_nu = self.couplings['angular'][mu][ell_nu][m_nu+ell_nu]

        u_mu = self.disc.roots[ell_mu][n_mu-1]
        k_mu = self.disc.wavenumbers[ell_mu][n_mu-1]
        k_nu = self.disc.wavenumbers[ell_nu][n_nu-1]

        if not callable(self.selection) and not callable(self.weight) \
                and ell_mu == ell_nu:
            if n_mu == n_nu:
                shot_noise = rmax**3 * spherical_besselj(ell_mu+1, u_mu)**2 / 2
            else:
                shot_noise = 0.
        else:
            args = mu, nu, k_mu, k_nu
            kwargs = dict(selection=self.selection, weight=self.weight)
            shot_noise = rad_int(
                lambda r: _shot_noise_kernel(r, *args, **kwargs),
                rmax
            )

        shot_noise *= M_mu_nu / self.mean_density

        return shot_noise

    def two_point_covariance(self, pivot, part='both'):
        """2-point signal, shot noise or full covariance matrix for given
        pivot axis for unpacking indices.

        Parameters
        ----------
        pivot : {'natural', 'scale', 'lmn', 'lnm', 'nlm', 'ln', 'k'}
            Pivot axis order.
        part : {'both', 'signal', 'shotnoise'}, optional
            If ``'both'`` (default), compute the sum of the signal and shot
            noise parts.  If ``'signal'`` or ``'shotnoise'``, compute
            only the corresponding part.

        Returns
        -------
        two_point_covar : complex :class:`numpy.ndarray`
            2-point covariance matrix pivoted at given axis order.

        Raises
        ------
        ValueError
            If `part` is not a recognised 2-point covariance component.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index_vector = SphericalArray.build(disc=self.disc)\
                .unfold(pivot, return_only='index')

        if part == 'both':
            two_point_component = lambda mu, nu: \
                self.two_point_signal(mu, nu) \
                + self.two_point_shot_noise(mu, nu)
        elif part == 'signal':
            two_point_component = lambda mu, nu: \
                self.two_point_signal(mu, nu)
        elif part == 'shotnoise':
            two_point_component = lambda mu, nu: \
                self.two_point_shot_noise(mu, nu)
        else:
            raise ValueError(f"Invalid covariance part: {part}. ")

        dim_covar = len(index_vector)
        two_point_covar = np.zeros((dim_covar, dim_covar), dtype=complex)
        for row_idx in range(dim_covar):
            for col_idx in range(row_idx+1):
                mu, nu = index_vector[row_idx], index_vector[col_idx]
                two_point_covar[row_idx, col_idx] = two_point_component(mu, nu)

        triu_indices = np.triu_indices(dim_covar, k=1)
        two_point_covar[triu_indices] = two_point_covar.T[triu_indices]

        return two_point_covar
