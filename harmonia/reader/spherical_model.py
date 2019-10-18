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
include the following factors:

    * angular mask :math:`M(\hat{\mathbf{r}})`;
    * radial selection :math:`\phi(r)`, and radial weight :math:`w(r)` or
      its derivative :math:`w'(r)`;
    * linear bias evolution :math:`G(z) = b_1(z) / b_1(0)` normalised to
      unity at the current epoch, where :math:`b_1(z)` is the
      scale-independent linear bias;
    * clustering evolution, :math:`D(z)`, which is the linear growth factor
      normalised to unity at the current epoch;
    * linear growth rate evolution :math:`F(z) = f(z) / f_0` normalised to
      unity at the current epoch, where :math:`f_0 \equiv f(0)`;
    * Alcock--Paczynski distortion

      .. math::

          \gamma(z) =
              \frac{\operatorname{d}\!\tilde{r}(z)}{\operatorname{d}\!r(z)}
              \,,

      where :math:`\tilde{r} = \tilde{r}(z)` is the fiducial distance
      converted from redshift rather than from the true comoving
      distance--redshift correspondence :math:`z = z(r)`.

When using integration kernels that is a combination of functions of the
above, pass additional parameters not being directly integrated over by
redefining these functions with the `lambda` keyword.

Couplings
---------------------------------------------------------------------------

Coupling coefficients are computed by integrating the angular, radial and
RSD coupling kernels

.. math::

   \begin{align*}
       M_{\mu\nu} &= \int \operatorname{d}^2\!\hat{\mathbf{r}}
           Y_{\ell_\mu m_\mu}^*(\hat{\mathbf{r}})
           M(\hat{\mathbf{r}})
           Y_{\ell_\nu m_\nu}(\hat{\mathbf{r}}) \,, \\
       \Phi_{\mu\nu} &= \kappa_{\ell_\nu n_\nu}
           \int \operatorname{d}\!r r^2 w(\tilde{r})
           j_{\ell_\mu}(k_{\ell_\mu n_\mu} \tilde{r})
           j_{\ell_\nu}(k_{\ell_\nu n_\nu} r)
           G(z) D(z) \phi(r) \,, \\
       \Upsilon_{\mu\nu} &=
           \frac{\kappa_{\ell_\nu n_\nu}}{k_{\ell_\nu n_\nu}}
           \int \operatorname{d}\!r r^2
           \frac{\operatorname{d}\!}{\operatorname{d}\!\tilde{r}}
           \left[ w(\tilde{r}) j_{\ell_\mu}(k_{\ell_\mu n_\mu} \tilde{r})
               \right]
           j'_{\ell_\nu}(k_{\ell_\nu n_\nu} r)
           \gamma(z) F(z) D(z) \phi(r) \,,
   \end{align*}

over the spherical volume element, where :math:`k_{\ell n}` are the
discrete wavenumbers.

When there is no angular masking (i.e. :math:`M(\hat{\mathbf{r}})` is
constant), the coupling coefficients reduce to :math:`M_{\mu\nu} =
\delta_{\mu\nu}`; if in addition radial selection, weighting and
evolutionary effects are all absent and the distance--redshift conversion
is the cosmological one (i.e. no AP correction), the radial coupling
coefficients is equivalent to :math:`\Phi_{\mu\nu} = \delta_{\mu\nu}`.

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
        \right] \kappa_\sigma^{-1} P_\textrm{m,0}(k_\sigma) \,,

and the shot noise part

.. math::

    \left\langle \epsilon_\mu \epsilon_\nu \right\rangle =
        \frac{1}{\bar{n}} M_{\mu\nu} \int \operatorname{d}\!r r^2
        (w^2\phi)(r) j_\mu(r) j_\nu(r) \,,

where the scale-dependent bias :math:`b_0(k) = b_1(0) + f_\textrm{NL}
\Delta b(k)` includes the modification :math:`\Delta b(k)` due to
primordial non-Gaussianity :math:`f_\textrm{NL}`, computed at the current
epoch (see :mod:`~harmonia.cosmology.scale_dependence`);
:math:`P_\textrm{m,0}` is the matter power spectrum at the current epoch;
:math:`\kappa` denotes the normalisation coefficients (see
:class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`); and
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
from harmonia.collections.utils import mpi_compute
from harmonia.cosmology.scale_dependence import scale_modification


# KERNELS
# -------------------------------------------------------------------------

def _angular_kernel(theta, phi, mu, nu, mask=None):
    r"""Evaluate the angular coupling kernel.

    Parameters
    ----------
    theta, phi : float, array_like
        Angular coordinates :math:`\theta`, :math:`\phi`.
    mu, nu : tuple or list of int
        Coefficient triplet index.
    mask : callable or None, optional
        Mask as a function of angular coordinates (default is `None`).

    Returns
    -------
    kernel : complex :class:`numpy.ndarray`
        Angular coupling kernel value.

    """
    kernel = np.conj(spherical_harmonic(mu[0], mu[1], theta, phi)) \
        * spherical_harmonic(nu[0], nu[1], theta, phi)

    if callable(mask):
        kernel *= mask(theta, phi)
    else:
        warnings.warn(
            "Angular kernel evaluation is redundant as `mask` is None. ",
            RuntimeWarning
        )

    return kernel


def _radial_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
                   bias_evolution=None, clustering_evolution=None,
                   z_from_r=None, chi_of_z=None):
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
    bias_evolution, clustering_evolution : callable or None, optional
        Bias and clustering evolution as a function of redshift normalised
        to unity at the current epoch (default is `None`).
    z_from_r : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default is
        `None`).
    chi_of_z : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is
        `None`).

    Returns
    -------
    kernel : float :class:`numpy.ndarray`
        Radial coupling kernel value.

    Raises
    ------
    TypeError
        If `z_from_r` is not callable when any of `bias_evolution`,
        `clustering_evolution` and `chi_of_z` is.

    """
    if not callable(chi_of_z):
        r_tilde = r
    else:
        if callable(z_from_r):
            r_tilde = chi_of_z(z_from_r(r))
        else:
            raise TypeError("`z_from_r` must be callable if `chi_of_z` is. ")

    kernel = spherical_besselj(mu[0], k_mu*r_tilde) \
        * spherical_besselj(nu[0], k_nu*r)

    if callable(selection):
        kernel *= selection(r)
    if callable(weight):
        kernel *= weight(r_tilde)
    if callable(clustering_evolution):
        if not callable(z_from_r):
            raise TypeError(
                "`z_from_r` must be callable if `clustering_evolution` is. "
            )
        kernel *= clustering_evolution(z_from_r(r))
    if callable(bias_evolution):
        if not callable(z_from_r):
            raise TypeError(
                "`z_from_r` must be callable if `bias_evolution` is. "
            )
        kernel *= bias_evolution(z_from_r(r))

    return kernel


def _RSD_kernel(r, mu, nu, k_mu, k_nu, selection=None, weight=None,
                weight_derivative=None, growth_evolution=None,
                clustering_evolution=None, z_from_r=None, chi_of_z=None,
                AP_distortion=None):
    """Evaluate the RSD coupling kernel.

    Parameters
    ----------
    r : float, array_like
        Radial coordinate.
    mu, nu : tuple or list of int
        Coefficient triplet index.
    k_mu, k_nu : float
        Discrete wave number corresponding to index `mu` or `nu`.
    selection, weight, weight_derivative : callable or None, optional
        Selection, weight or weight derivative as a function of the radial
        coordinate (default is `None`).
    growth_evolution, clustering_evolution : callable or None, optional
        Growth rate evolution or clustering evolution as a function of
        redshift normalised to unity at the current epoch (default is
        `None`).
    z_from_r : callable or None, optional
        Cosmological comoving distance-to-redshift conversion (default is
        `None`).
    chi_of_z : callable or None, optional
        Fiducial comoving redshift-to-distance conversion (default is
        `None`).
    AP_distortion : callable or None, optional
        AP distortion as a function of redshift (default is `None`).

    Returns
    -------
    kernel : float, array_like
        RSD coupling kernel value.

    Raises
    ------
    TypeError
        If `z_from_r` is not callable when any of `growth_evolution`,
        `clustering_evolution` and `chi_of_z` is.
    TypeError
        If `weight` is not callable when `weight_derivative` is, and vice
        versa.
    TypeError
        If `chi_of_z` is not callable when `AP_distortion` is, and vice
        versa.

    """
    require_z_from_r = any(
        [
            callable(growth_evolution),
            callable(clustering_evolution),
            callable(chi_of_z),
        ]
    )
    if not callable(z_from_r) and require_z_from_r:
        raise TypeError(
            "`z_from_r` must be callable if any of "
            "`growth_evolution`, `clustering_evolution` and `chi_of_z` is. "
        )

    apply_AP = callable(chi_of_z) and callable(AP_distortion)
    if (callable(chi_of_z) or callable(AP_distortion)) and not apply_AP:
        warnings.warn(
            "One of `chi_of_z` and `AP_distortion` is not callable, "
            "so both are set to None and no AP correction is applied. ",
            RuntimeWarning
        )

    apply_weight = callable(weight) and callable(weight_derivative)
    if (callable(weight) or callable(weight_derivative)) and not apply_weight:
        raise TypeError(
            "`weight` and `weight_derivative` must both be callable "
            "if either is. "
        )

    kernel = spherical_besselj(nu[0], k_nu*r, derivative=True)

    if not apply_AP:
        r_tilde = r
    else:
        r_tilde = chi_of_z(z_from_r(r))
        kernel *= AP_distortion(z_from_r(r))

    if callable(selection):
        kernel *= selection(r)

    if not apply_weight:
        kernel *= k_mu * spherical_besselj(mu[0], k_mu*r, derivative=True)
    else:
        kernel *= weight_derivative(r_tilde) \
            * spherical_besselj(mu[0], k_mu*r_tilde) \
            + k_mu * weight(r_tilde) \
            * spherical_besselj(mu[0], k_mu*r_tilde, derivative=True)

    if callable(growth_evolution):
        kernel *= growth_evolution(z_from_r(r))
    if callable(clustering_evolution):
        kernel *= clustering_evolution(z_from_r(r))

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
        warnings.warn("Shot noise evaluation is redundant. ", RuntimeWarning)

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
        mandatory keys: ``'z_from_r'`` for cosmological comoving
        distance-to-redshift conversion, ``'chi_of_z'`` for fiducial
        redshift-to-comoving distance conversion, ``'bias_evolution'``,
        ``'growth_evolution'`` and ``'clustering_evolution'`` for bias,
        growth rate and clustering evolution as functions of the redshift
        normalised to unity today, ``'AP_distortion'`` for AP distortion.
        Default is `None`.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

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
    z_from_r : callable or None
        Cosmological comoving distance-to-redshift conversion.
    chi_of_z : callable or None
        Fiducial comoving redshift-to-distance conversion.
    clustering_evolution : callable or None
        Clustering evolution as a function of redshift normalised to unity
        at the current epoch.
    bias_evolution : callable or None
        Bias evolution as a function of redshift normalised to unity
        at the current epoch.
    growth_evolution : callable or None
        Growth rate evolution as a function of redshift normalised to unity
        at the current epoch.
    AP_distortion : callable or None
        AP distortion as a function of redshift normalised to unity
        at the current epoch.

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

    _all_specs_attr = {
        "survey_specs": (
            'mask',
            'selection',
            'weight',
            'weight_derivative',
        ),
        "cosmo_specs": (
            'z_from_r',
            'chi_of_z',
            'clustering_evolution',
            'bias_evolution',
            'growth_evolution',
            'AP_distortion',
        ),
    }

    def __init__(self, disc, survey_specs=None, cosmo_specs=None, comm=None,
                 logger=None):

        if logger is None:
            self._logger = logging.getLogger("Couplings")

        self.comm = comm
        self.disc = disc

        for specs_type, specs_attrs in self._all_specs_attr.items():
            specs = locals()[specs_type]
            specs_var_str = "`" + specs_type + "`"
            if isinstance(specs, dict):
                try:
                    for func_attr in specs_attrs:
                        setattr(self, func_attr, specs[func_attr])
                        attr_func = getattr(self, func_attr)
                        if attr_func is not None and not callable(attr_func):
                            raise TypeError(
                                f"{specs_var_str} {func_attr} value "
                                f"must be None or callable. "
                            )
                except KeyError as missing_func_attr:
                    raise KeyError(
                        f"{specs_var_str} key {missing_func_attr} is missing. "
                    )
            elif specs is None:
                for func_attr in specs_attrs:
                    setattr(self, func_attr, None)

    def coupling_coefficient(self, mu, nu, coupling_type):
        r"""Evaluate couplings at specified indices.

        Parameters
        ----------
        mu, nu : tuple or list of int
            Coefficient triplet index.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.

        Returns
        -------
        coupling_coeff : float or complex :class:`numpy.ndarray`
            Coupling coefficient of given type for specified indices.

        """
        coupling_type = self._alias(coupling_type)

        _info_msg = "{} coupling for: {} and {}".format(
            coupling_type.replace("'", ""),
            str(mu).replace("'", ""),
            str(nu).replace("'", ""),
        )

        if self.comm is None or self.comm.rank == 0:
            self._logger.debug("Computing %s. ", _info_msg)

        if coupling_type != 'angular':
            k_mu = self.disc.wavenumbers[mu[0]][mu[-1]-1]
            k_nu = self.disc.wavenumbers[nu[0]][nu[-1]-1]
            kappa_nu = self.disc.normalisations[nu[0]][nu[-1]-1]

            attrs = [
                'selection',
                'weight',
                'clustering_evolution',
                'z_from_r',
                'chi_of_z',
            ]

        if coupling_type == 'angular':
            trivial_case = not callable(self.mask)
            if trivial_case:  # Kronecker delta
                coupling_coeff = complex(mu[0] == nu[0] and mu[1] == nu[1])

            else:
                coupling_coeff = ang_int(
                    lambda theta, phi: \
                        _angular_kernel(theta, phi, mu, nu, mask=self.mask)
                )
        elif coupling_type == 'radial':
            attrs.extend(['bias_evolution'])
            func_attrs = {attr: getattr(self, attr) for attr in attrs}

            trivial_case = (mu[0] == nu[0]) and not any(
                [callable(func) for attr, func in func_attrs.items()]
            )
            if trivial_case:  # Kronecker delta
                coupling_coeff = float(mu[-1] == nu[-1])
            else:
                coupling_coeff = kappa_nu * rad_int(
                    lambda r: \
                        _radial_kernel(r, mu, nu, k_mu, k_nu, **func_attrs),
                    self.disc.attrs['boundary_radius']
                )
        elif coupling_type == 'RSD':
            attrs.extend(
                ['weight_derivative', 'growth_evolution', 'AP_distortion']
            )
            func_attrs = {attr: getattr(self, attr) for attr in attrs}

            coupling_coeff = kappa_nu / k_nu * rad_int(
                lambda r: _RSD_kernel(r, mu, nu, k_mu, k_nu, **func_attrs),
                self.disc.attrs['boundary_radius']
            )

        if self.comm is None or self.comm.rank == 0:
            self._logger.debug("Computed %s.", _info_msg)

        return coupling_coeff

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

        """
        coupling_type = self._alias(coupling_type)

        _info_msg = "all {} couplings".format(coupling_type.replace("'", ""))

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("Computing %s.", _info_msg)

        couplings_component = {}
        if coupling_type == 'angular':  # exploiting `m`-parity
            sigma_gen = lambda ell: [(ell, m, None) for m in range(-ell, 1)]
            for ell in self.disc.degrees:
                ell_component = np.array(
                    [
                        self.coupling_coefficient(mu, sigma, coupling_type)
                        for sigma in sigma_gen(ell)
                    ]
                )
                ell_component_parity = np.conj(
                    np.power(-1, np.arange(1, ell+1)) *
                    np.flip(ell_component[:-1])
                )
                couplings_component[ell] = np.concatenate(
                    (ell_component, ell_component_parity)
                )
        else:
            sigma_gen = lambda ell: [
                (ell, None, n) for n in range(1, self.disc.depths[ell]+1)
            ]
            for ell in self.disc.degrees:
                ell_component = np.array(
                    [
                        self.coupling_coefficient(mu, sigma, coupling_type)
                        for sigma in sigma_gen(ell)
                    ]
                )
                couplings_component[ell] = ell_component

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("Computed %s.", _info_msg)

        return couplings_component

    def couplings(self, coupling_type):
        r"""Compile all coupling coefficients of a given type as a sequence
        iterated through the first triplet index ordered as specified.

        This returns a dictionary whose keys are all the triplet indices,
        each with a value corresponding to the coefficients returned by
        a call of
        :meth:`~.reader.spherical_model.Couplings.couplings_fixed_index`
        for the specified coupling type and that triplet index.

        Parameters
        ----------
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.

        Returns
        -------
        sequenced_couplings : dict of {tuple: |nested_dict|}
            Sequence of coupling coefficients over the first triplet index.


        .. |nested_dict| replace::

            dict of {int: :class:`numpy.ndarray`}

        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index_vector = SphericalArray.build(disc=self.disc)\
                .unfold('natural', return_only='index')

        if self.comm is not None:
            coeff_processor = lambda mu: \
                self.couplings_fixed_index(mu, coupling_type=coupling_type)
            coeff_vector = \
                mpi_compute(index_vector, coeff_processor, self.comm)

            if self.comm.rank == 0:
                sequenced_couplings = dict(zip(index_vector, coeff_vector))
                sequenced_couplings = \
                    self.comm.bcast(sequenced_couplings, root=0)
        else:
            coeff_vector = [
                self.couplings_fixed_index(mu, coupling_type=coupling_type)
                for mu in index_vector
            ]
            sequenced_couplings = dict(zip(index_vector, coeff_vector))

        return sequenced_couplings

    @staticmethod
    def _alias(coupling_type):

        if coupling_type.lower().startswith('a'):
            return 'angular'
        if coupling_type.lower().startswith('rad'):
            return 'radial'
        if coupling_type.lower().startswith('rsd'):
            return 'RSD'
        raise ValueError(
            "`coupling_type` can only be: 'angular', 'radial' or 'RSD'. "
        )


# 2-Point Correlators
# -----------------------------------------------------------------------------

class TwoPointFunction(Couplings):
    r"""Compute 2-point function values for given survey and cosmological
    specifications from a biased power spectrum model, linear growth rate
    and local primordial non-Gaussianity.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    b_const : float
        Constant linear bias of the tracer particles at the current epoch.
    f_0 : float or None, optional
        Linear growth rate at the current epoch.  If `None` (default), this
        is set to zero and RSD calculations are neglected.
    power_spectrum : callable or None
        Linear matter power spectrum model.
    cosmo : :class:`nbodykit.cosmology.Cosmology` *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating
        scale-dependent bias.
    survey_specs : dict of {str: callable or None} or None, optional
        Survey specification functions accessed with the following
        mandatory keys: ``'mask'`` for angular mask, and ``'selection'``
        and ``'weight'`` for radial selection and weighting,
        ``'weight_derivative'`` for the derivative function of radial
        weighting.  Default is `None`.  Ignored if `cosmo` is `None`.
    cosmo_specs : dict of {str: callable or None} or None, optional
        Cosmological specification functions accessed with the following
        mandatory keys: ``'z_from_r'`` for cosmological comoving
        distance-to-redshift conversion, ``'chi_of_z'`` for fiducial
        redshift-to-comoving distance conversion, ``'bias_evolution'``,
        ``'growth_evolution'`` and ``'clustering_evolution'`` for bias,
        growth rate and clustering evolution as functions of the redshift
        normalised to unity today, ``'AP_distortion'`` for AP distortion.
        Default is `None`.
    couplings : dict of {str: dict of {tuple: dict}} or None, optional
        Pre-computed couplings (default is `None`).
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
    matter_power_spectrum : |linear_power|
        Linear matter power spectrum model at the current epoch (in cubic
        Mpc/h).
    cosmo : :class:`nbodykit.cosmology.Cosmology` *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating
        scale-dependent bias.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.


    .. |linear_power| replace::

         :class:`nbodykit.cosmology.power.linear.LinearPower`

    """

    _logger = logging.getLogger("TwoPointFunction")
    _CURRENT_Z = 0.

    def __init__(self, disc, nbar, b_const, f_0=None, power_spectrum=None,
                 cosmo=None, survey_specs=None, cosmo_specs=None,
                 couplings=None, comm=None):

        super().__init__(
            disc,
            survey_specs=survey_specs,
            cosmo_specs=cosmo_specs,
            comm=comm,
            logger=self._logger
        )

        self.mean_density = nbar
        self.bias_const = b_const

        if cosmo is None:
            self.growth_rate = f_0
            self.matter_power_spectrum = power_spectrum
        else:
            # Overide non-None `f_0` value for model consistency.
            if f_0 is not None:
                self.growth_rate = \
                    cosmo.scale_independent_growth_rate(self._CURRENT_Z)
            else:
                self.growth_rate = None
            self.matter_power_spectrum = cosmology.LinearPower(
                cosmo,
                redshift=self._CURRENT_Z,
                transfer='CLASS'
            )

        self.cosmo = cosmo
        self._couplings = couplings
        self._mode_powers = None
        self._mode_scale_modifications = None

    @property
    def couplings(self):
        """Coupling coefficients of all types for all indices.

        Returns
        -------
        dict of {str :code:`:` dict}
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
            self._couplings[coupling_type] = super().couplings(coupling_type)

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("Coupling coefficients compiled. ")

        return self._couplings

    @property
    def mode_powers(self):
        """Power of the underlying matter distribution at the discretised
        Fourier modes.

        Returns
        -------
        dict of {int: float :class:`numpy.ndarray``}
            Mode power unmodulated by tracer bias.

        """
        if self._mode_powers is not None:
            return self._mode_powers

        self._mode_powers = {}
        for ell in self.disc.degrees:
            self._mode_powers[ell] = \
                self.matter_power_spectrum(self.disc.wavenumbers[ell])

        return self._mode_powers

    @property
    def mode_scale_modifications(self):
        """Scale-dependent modification to constant linear bias at
        discretised Fourier modes to be modulated by local primordial
        non-Gaussianity and tracer parameter.

        Returns
        -------
        dict of {int: float :class:`numpy.ndarray`}
            Mode scale-dependent modification.

        """
        if self._mode_scale_modifications is not None:
            return self._mode_scale_modifications

        scale_modification_kernel = \
            scale_modification(self.cosmo, self._CURRENT_Z)

        self._mode_scale_modifications = {}
        for ell in self.disc.degrees:
            self._mode_scale_modifications[ell] = \
                scale_modification_kernel(self.disc.wavenumbers[ell])

        return self._mode_scale_modifications

    def two_point_signal(self, mu, nu, f_nl=None, tracer_parameter=1.):
        """Signal 2-point function for given triplet indices.

        Parameters
        ----------
        mu, nu : tuple or list of int
            Coefficient triplet index.
        f_nl : float or None, optional
            Local primordial non-Gaussianity parameter (default is `None`).
        tracer_parameter : float, optional
            Tracer species--dependent parameter for bias modulation
            (default is 1.).

        Returns
        -------
        signal : complex
            Cosmological signal 2-point function value for given triplet
            indices.

        """
        f_0 = self.growth_rate

        couplings = self.couplings

        M_mu, M_nu = couplings['angular'][mu], couplings['angular'][nu]
        Phi_mu, Phi_nu = couplings['radial'][mu], couplings['radial'][nu]
        if f_0 is not None:
            Upsilon_mu, Upsilon_nu = couplings['RSD'][mu], couplings['RSD'][nu]

        kappa = self.disc.normalisations
        b_const = self.bias_const
        A_k = self.mode_scale_modifications
        p_k = self.mode_powers

        signal = 0.
        for ell, nmax in zip(self.disc.degrees, self.disc.depths):
            print(ell, M_mu[ell].shape)
            angular_sum = np.sum(
                [
                    M_mu[ell][m_idx] * np.conj(M_nu[ell][m_idx])
                    for m_idx in range(0, 2*ell+1)
                ]
            )

            b_0_k = b_const * np.ones(nmax)
            if f_nl is not None:
                b_0_k += f_nl * (b_const - tracer_parameter) * A_k[ell]

            if f_0 is None:
                radial_sum = np.sum(
                    Phi_mu[ell] * Phi_nu[ell]
                    * b_0_k**2 * p_k[ell] / kappa[ell]
                )
            else:
                radial_sum = np.sum(
                    (b_0_k * Phi_mu[ell] + f_0 * Upsilon_mu[ell])
                    * (b_0_k * Phi_nu[ell] + f_0 * Upsilon_nu[ell])
                    * p_k[ell] / kappa[ell]
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

        M_mu_nu = (self.couplings['angular'][mu])[ell_nu][m_nu+ell_nu]

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

    def two_point_covariance(self, pivot, part='both', diag=False,
                             f_nl=None, tracer_parameter=1.):
        """2-point signal, shot noise or full covariance matrix for given
        pivot axis for unpacking indices.

        Parameters
        ----------
        pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Pivot axis order for unpacking indices.
        part : {'both', 'signal', 'shotnoise'}, optional
            If ``'both'`` (default), compute the sum of the signal and shot
            noise parts.  If ``'signal'`` or ``'shotnoise'``, compute
            only the corresponding part.
        diag : bool, optional
            If `True` (default is `False`), only compute the diagonal
            entries of the covariance matrix.
        f_nl : float or None, optional
            Local primordial non-Gaussianity parameter (default is `None`).
        tracer_parameter : float, optional
            Tracer species--dependent parameter for bias modulation
            (default is 1.).

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
                self.two_point_signal(mu, nu, f_nl=None, tracer_parameter=1.) \
                + self.two_point_shot_noise(mu, nu)
        elif part == 'signal':
            two_point_component = lambda mu, nu: \
                self.two_point_signal(mu, nu, f_nl=None, tracer_parameter=1.)
        elif part == 'shotnoise':
            two_point_component = lambda mu, nu: \
                self.two_point_shot_noise(mu, nu)
        else:
            raise ValueError(f"Invalid covariance part: {part}. ")

        dim_covar = len(index_vector)
        two_point_covar = np.zeros((dim_covar, dim_covar), dtype=complex)
        if diag:
            for diag_idx in range(dim_covar):
                mu = index_vector[diag_idx]
                two_point_covar[diag_idx, diag_idx] = \
                    two_point_component(mu, mu)
        else:
            for row_idx in range(dim_covar):
                for col_idx in range(row_idx+1):
                    mu, nu = index_vector[row_idx], index_vector[col_idx]
                    two_point_covar[row_idx, col_idx] = \
                        two_point_component(mu, nu)

            triu_indices = np.triu_indices(dim_covar, k=1)
            two_point_covar[triu_indices] = two_point_covar.T[triu_indices]

        return two_point_covar
