r"""
Spherical Fourier-space model (:mod:`~harmonia.reader.spherical_model`)
===========================================================================

Compute Fourier-space :math:`n`-point functions in the spherical basis.
The index for each Fourier coefficient is a triplet :math:`(\ell_\mu,
m_\mu, n_\mu)`, i.e. ``(mu[0], mu[1], mu[2])``.

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
    * linear bias evolution :math:`G(z,k) = b(z,k) / b(z_*,k)` normalised
      to unity at the current epoch :math:`z_*`, where :math:`b(z,k)` is
      the scale-dependent linear bias;
    * clustering evolution, :math:`D(z)`, which is the linear growth factor
      normalised to unity at the :math:`z = 0` epoch;
    * linear growth rate evolution :math:`F(z) = f(z) / f_*` normalised to
      unity at the current epoch :math:`z_*`, where :math:`f_* \equiv
      f(z_*)`;
    * Alcock--Paczynski differential distortion

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
       M_{\mu\nu} &= \int \operatorname{d}^2\!\hat{\mathbf{r}} \,
           Y_{\ell_\mu m_\mu}^*(\hat{\mathbf{r}})
           M(\hat{\mathbf{r}})
           Y_{\ell_\nu m_\nu}(\hat{\mathbf{r}}) \,, \\
       \Phi_{\mu\nu} &= \kappa_{\ell_\nu n_\nu}
           \int \operatorname{d}\!r \, r^2 w(\tilde{r})
           j_{\ell_\mu}(k_{\ell_\mu n_\mu} \tilde{r})
           j_{\ell_\nu}(k_{\ell_\nu n_\nu} r)
           G(z, k_{\ell_\nu n_\nu}) D(z) \phi(r) \,, \\
       \Upsilon_{\mu\nu} &=
           \frac{\kappa_{\ell_\nu n_\nu}}{k_{\ell_\nu n_\nu}}
           \int \operatorname{d}\!r \, r^2
           \frac{\operatorname{d}\!}{\operatorname{d}\!\tilde{r}}
           \left[ w(\tilde{r}) j_{\ell_\mu}(k_{\ell_\mu n_\mu} \tilde{r})
               \right]
           j'_{\ell_\nu}(k_{\ell_\nu n_\nu} r)
           \gamma(z) F(z) D(z) \phi(r) \,,
   \end{align*}

over the spherical volume element, where :math:`k_{\ell n}` is the
discrete wavenumber.

When there is no angular masking (i.e. :math:`M(\hat{\mathbf{r}})` is
constant), the coupling coefficients reduce to :math:`M_{\mu\nu} =
\delta_{\mu\nu}`; if in addition radial selection, weighting and
evolutionary effects are all absent and the distance--redshift conversion
is the cosmological one (i.e. no AP correction), then
:math:`M_{\mu\nu} \Phi_{\mu\nu} = \delta_{\mu\nu}`.

.. autosummary::

    Couplings


2-point functions
---------------------------------------------------------------------------

2-point functions are computed from couplings as a sum of the signal part

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

where the scale-dependent bias :math:`b(z,k) = b_1(z) + f_\textrm{NL}
\Delta b(z,k)` includes the modification :math:`\Delta b` due to local
primordial non-Gaussianity :math:`f_\textrm{NL}`, computed at the current
epoch (see :mod:`~harmonia.cosmology.scale_dependence`);
:math:`P_{\textrm{m}*}` is the matter power spectrum at the current epoch
:math:`z_*`; :math:`\kappa` denotes the normalisation coefficients (see
:class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`); and
:math:`j_\mu(r) \equiv j_{\ell_\mu}(k_{\ell_\mu n_\mu} r)`.

.. autosummary::

    TwoPointFunction

|

"""
import logging
import warnings
from collections import defaultdict
from itertools import product

import numpy as np
from nbodykit.lab import cosmology
from scipy.integrate import IntegrationWarning

from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic
from harmonia.algorithms.integration import \
    angular_spherical_integral as ang_int
from harmonia.algorithms.integration import \
    radial_spherical_integral as rad_int
from harmonia.algorithms.arrays import SphericalArray
from harmonia.collections.utils import mpi_compute, progress_status
from harmonia.cosmology.scale_dependence import scale_dependence_modification

__all__ = ['Couplings', 'TwoPointFunction']


class CouplingCoefficientWarning(UserWarning):
    """Coupling coefficient warning from poor integration.

    """
    pass


# KERNELS
# -----------------------------------------------------------------------------

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
            "Angular kernel evaluation is redundant as `mask` is None. "
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
        warnings.warn("Shot noise evaluation is redundant. ")

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

    _logger = logging.getLogger("Couplings")

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

    def __init__(self, disc, survey_specs=None, cosmo_specs=None, comm=None):

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
                                "must be None or callable. "
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
                with warnings.catch_warnings(record=True) as any_warning:
                    coupling_coeff = ang_int(
                        lambda theta, phi: _angular_kernel(
                            theta, phi, mu, nu, mask=self.mask
                        )
                    )
                if any_warning and not np.isclose(coupling_coeff, 0.):
                    warnings.warn(
                        "Angular integration warning for index pair "
                        "{} and {}: {}.\n"
                        .format(mu, nu, any_warning[-1].message),
                        category=CouplingCoefficientWarning
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
                with warnings.catch_warnings(record=True) as any_warning:
                    coupling_coeff = kappa_nu * rad_int(
                        lambda r: _radial_kernel(
                            r, mu, nu, k_mu, k_nu, **func_attrs
                        ),
                        self.disc.attrs['boundary_radius']
                    )
                if any_warning:
                    warnings.warn(
                        "Radial integration warning for index pair "
                        "{} and {}: {}.\n"
                        .format(mu, nu, any_warning[-1].message),
                        category=CouplingCoefficientWarning
                    )
        elif coupling_type == 'RSD':
            attrs.extend(
                ['weight_derivative', 'growth_evolution', 'AP_distortion']
            )
            func_attrs = {attr: getattr(self, attr) for attr in attrs}

            with warnings.catch_warnings(record=True) as any_warning:
                coupling_coeff = kappa_nu / k_nu * rad_int(
                    lambda r: _RSD_kernel(r, mu, nu, k_mu, k_nu, **func_attrs),
                    self.disc.attrs['boundary_radius']
                )
            if any_warning:
                warnings.warn(
                    "RSD integration warning for index pair "
                    "{} and {}: {}.\n"
                    .format(mu, nu, any_warning[-1].message),
                    category=CouplingCoefficientWarning
                )

        if self.comm is None or self.comm.rank == 0:
            self._logger.debug("Computed %s.", _info_msg)

        return coupling_coeff

    def _couplings_fixed_index(self, mu, coupling_type):
        r"""Compute coupling coefficients with the first triplet index
        fixed.

        Notes
        -----
        This function computes coupling coefficients of the form
        :math:`C_{a_\mu b_\mu a_\sigma b_\sigma}` where the triplet index
        :math:`\mu` is fixed, and compiles their values as a vector by
        iterating through the triplet index :math:`\sigma`.  For example,

        ::

            Couplings._couplings_fixed_index(mu, 'angular')

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

        if coupling_type == 'angular':
            sigma_gen = lambda ell: [
                (ell, m, None) for m in range(-ell, ell+1)
            ]
        else:
            sigma_gen = lambda ell: [
                (ell, None, n) for n in range(1, self.disc.depths[ell]+1)
            ]

        couplings_component = {}
        for ell in self.disc.degrees:
            couplings_component[ell] = np.array(
                [
                    self.coupling_coefficient(mu, sigma, coupling_type)
                    for sigma in sigma_gen(ell)
                ]
            )

        return couplings_component

    def compile_couplings(self, coupling_type):
        r"""Compile all coupling coefficients of a given type as a sequence
        iterated through the first coupling index in the discrete spectrum.

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
        coupling_type = self._alias(coupling_type)

        _info_msg = "all {} couplings".format(coupling_type.replace("'", ""))

        index_vector = self._reduced_index_vector(coupling_type)

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("Computing %s. ", _info_msg)
        if self.comm is not None:
            self.comm.Barrier()

        with warnings.catch_warnings(record=True) as captured_warnings:
            if self.comm is None:
                coeff_vector = []
                for ind_idx, mu in enumerate(index_vector):
                    coeff_vector.append(
                        self._couplings_fixed_index(
                            mu, coupling_type=coupling_type
                        )
                    )
                    progress_status(
                        ind_idx, len(index_vector), self._logger,
                        process_name=f"{coupling_type} coupling evaluation"
                    )
            else:
                coeff_processor = lambda mu: self._couplings_fixed_index(
                    mu, coupling_type=coupling_type
                )
                coeff_vector = mpi_compute(
                    index_vector, coeff_processor, self.comm,
                    logger=self._logger,
                    process_name=f"{coupling_type} coupling evaluation"
                )

        unique_warning_msgs = set(map(
            lambda warning_obj: warning_obj.message, captured_warnings
        ))
        for msg in unique_warning_msgs:
            warnings.warn(msg, IntegrationWarning)

        sequenced_couplings = dict(zip(index_vector, coeff_vector))

        if self.comm is not None:
            self.comm.Barrier()
        if self.comm is None or self.comm.rank == 0:
            self._logger.info("Computed %s. ", _info_msg)

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
            f"Unrecognised `coupling_type`: {coupling_type}. "
        )

    def _reduced_index_vector(self, coupling_type):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index_vector = SphericalArray\
                .build(disc=self.disc)\
                .unfold('natural', return_only='index')

        if coupling_type == 'angular':
            operatable_index_vector = map(
                lambda tup: (tup[0], tup[1], None), index_vector
            )
        else:
            operatable_index_vector = map(
                lambda tup: (tup[0], None, tup[2]), index_vector
            )

        # ``Sorted`` orders ``set`` output.
        return sorted(list(set(operatable_index_vector)))


# 2-Point Correlators
# -----------------------------------------------------------------------------

class TwoPointFunction(Couplings):
    """Compute 2-point function values for given survey and cosmological
    specifications.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    redshift : float, optional
        Current redshift at which 2-point functions are modelled (default
        is 0.).
    growth_rate : float or None, optional
        Linear growth rate at the current epoch.  If `None` (default), an
        attempt is made to compute its value from `redshift` and `cosmo`;
        should `cosmo` be `None`, it is set to 0, in which case no RSD
        calculations are invoked.
    power_spectrum : callable or None
        Linear matter power spectrum model at the current epoch.
    cosmo : |cosmo| *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating the
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
        Pre-computed couplings (default is `None`) for cosmology and
        redshift consistent with the other arguments.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.

    Attributes
    ----------
    redshift : float
        Current redshift at which the 2-point functions are modelled.
    growth_rate : float or int
        Linear growth rate at the current epoch.  This is 0 if RSD effects
        are ignored.
    matter_power_spectrum : callable
        Linear matter power spectrum model at the current epoch (in cubic
        Mpc/:math:`h`).
    cosmo : |cosmo| *or None, optional*
        Cosmological model used to produce a power spectrum model, linear
        growth rate and the transfer function for calculating the
        scale-dependent bias.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator.  If `None` (default), no multiprocessing
        is invoked.


    .. todo::

        Fully include evolution and AP effects by resolving the conflicts
        between attributes set by `cosmo_specs` and derived by `cosmo`.


    .. |cosmo| replace::

        :class:`nbodykit.cosmology.cosmology.Cosmology`

    """

    _logger = logging.getLogger("TwoPointFunction")

    def __init__(self, disc, redshift=0., growth_rate=None,
                 power_spectrum=None, cosmo=None, survey_specs=None,
                 cosmo_specs=None, couplings=None, comm=None):

        super().__init__(
            disc, survey_specs=survey_specs, cosmo_specs=cosmo_specs, comm=comm
        )

        self.redshift = redshift
        self.growth_rate = growth_rate
        self.matter_power_spectrum = power_spectrum

        if cosmo is None:
            if self.growth_rate is None:
                self.growth_rate = 0
            if self.matter_power_spectrum is None:
                raise ValueError(
                    "power_spectrum` cannot be None when `cosmo` is None. "
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
            if self.growth_rate is None:
                self.growth_rate = cosmo.scale_independent_growth_rate(
                    self.redshift
                )
            elif self.growth_rate:
                cosmo_growth_rate = cosmo.scale_independent_growth_rate(
                    self.redshift
                )
                if not np.isclose(self.growth_rate, cosmo_growth_rate):
                    warnings.warn(
                        "Input `growth_rate` deviates from "
                        "the general relativistic prediction for "
                        "input `cosmo` model: "
                        "input value {}, predicted value {}. "
                        .format(self.growth_rate, cosmo_growth_rate)
                    )
        self.cosmo = cosmo

        self._natural_indices = SphericalArray.build(disc=self.disc)
        self._couplings = couplings
        self._mode_powers_ = None
        self._mode_scale_dependence_modifications_ = None
        self._fixed_angular_sums_ = None
        self._fixed_shot_noise_ = None

    @property
    def couplings(self):
        """Coupling coefficients of all types for all indices.

        Returns
        -------
        dict of {str :code:`:` dict}
            Coupling coefficients as a dictionary for coupling types
            each with a sub-dictionary for all triplet indices.

        Raises
        ------
        ValueError
            If externally set non-NoneType :attr:`couplings` do not match
            `cosmo_specs` or `growth_rate`, or is missing ``'radial'`` key
            or ``'radial'``-key value is `None`.

        """
        if self._couplings is not None:
            couplings_missing = (
                self.mask is not None
                and (
                    'angular' not in self._couplings
                    or self._couplings['angular'] is None
                )
            ) or (
                bool(self.growth_rate)
                and (
                    'RSD' not in self._couplings
                    or self._couplings['RSD'] is None
                )
            ) or (
                'radial' not in self._couplings \
                or self._couplings['radial'] is None
            )
            if couplings_missing:
                raise ValueError(
                    "Externally loaded `couplings` mismatch "
                    "non-NoneType `cosmo_spec` or `growth_rate`. "
                )
            return self._couplings

        self._couplings = {'radial': super().compile_couplings('radial')}
        if self.mask:
            self._couplings['angular'] = super().compile_couplings('angular')
        else:
            self._couplings['angular'] = None

        if self.growth_rate:
            self._couplings['RSD'] = super().compile_couplings('RSD')
        else:
            self._couplings['RSD'] = None

        if self.comm is None or self.comm.rank == 0:
            self._logger.info("Relevant coupling coefficients compiled. ")

        return self._couplings

    def two_point_signal(self, mu, nu, b_1, f_nl=None, tracer_parameter=1.):
        """Compute signal 2-point function for given triplet indices with
        or without scale-dependence modification by local primordial
        non-Gaussianity.

        Parameters
        ----------
        mu, nu : tuple or list of int
            Coefficient triplet index.
        b_1 : float
            Scale-independent linear bias of the tracer particles at the
            current epoch.
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
        angular_reduction = (self.couplings['angular'] is None)
        rsd_reduction = not bool(self.growth_rate)

        Phi_mu = self._access_couplings('radial', mu)
        Phi_nu = self._access_couplings('radial', nu)

        if not rsd_reduction:
            Upsilon_mu = self._access_couplings('RSD', mu)
            Upsilon_nu = self._access_couplings('RSD', nu)

        f = self.growth_rate
        p_k = self._mode_powers
        kappa = self.disc.normalisations

        signal = 0.
        if angular_reduction:
            trivial_case = (mu[:-1] != nu[:-1])
            if not trivial_case:
                ell = mu[0]  # equivalently nu[0]
                nmax = self.disc.depths[ell]

                b_k = b_1 * np.ones(nmax)
                if f_nl is not None:
                    b_k += f_nl * (b_1 - tracer_parameter) \
                        * self._mode_scale_dependence_modifications[ell]

                if rsd_reduction:
                    radial_sum = np.sum(
                        Phi_mu[ell] * Phi_nu[ell]
                        * b_k**2 * p_k[ell] / kappa[ell]
                    )
                else:
                    radial_sum = np.sum(
                        (b_k * Phi_mu[ell] + f * Upsilon_mu[ell])
                        * (b_k * Phi_nu[ell] + f * Upsilon_nu[ell])
                        * p_k[ell] / kappa[ell]
                    )

                signal += radial_sum
        else:
            for ell, nmax in zip(self.disc.degrees, self.disc.depths):

                angular_sum = self._access_angular_sums(ell, mu, nu)

                b_k = b_1 * np.ones(nmax)
                if f_nl is not None:
                    b_k += f_nl * (b_1 - tracer_parameter) \
                        * self._mode_scale_dependence_modifications[ell]

                if rsd_reduction:
                    radial_sum = np.sum(
                        Phi_mu[ell] * Phi_nu[ell]
                        * b_k**2 * p_k[ell] / kappa[ell]
                    )
                else:
                    radial_sum = np.sum(
                        (b_k * Phi_mu[ell] + f * Upsilon_mu[ell])
                        * (b_k * Phi_nu[ell] + f * Upsilon_nu[ell])
                        * p_k[ell] / kappa[ell]
                    )
                signal += angular_sum * radial_sum

        return signal

    def two_point_shot_noise(self, mu, nu, nbar, contrast=np.inf):
        """Compute shot noise 2-point function for given triplet indices.

        Parameters
        ----------
        mu, nu : tuple or list of int
            Coefficient triplet index.
        nbar : float
            Mean particle number density (in cubic :math:`h`/Mpc).
        contrast : float, optional
            Effectively downscale `nbar` by ``1 + 1/constrast``.  Default is
            ``numpy.inf``.

        Returns
        -------
        shot_noise : complex
            Shot noise 2-point function value for given triplet indices.

        """
        ell_mu, m_mu, n_mu = mu
        ell_nu, m_nu, n_nu = nu

        if self.couplings['angular'] is None:
            if ell_mu != ell_nu or m_mu != m_nu:
                return 0.j
            M_mu_nu = 1.
        else:
            M_mu_nu = \
                self._access_couplings('angular', mu)[ell_nu][m_nu+ell_nu]

        rmax = self.disc.attrs['boundary_radius']

        u_mu = self.disc.roots[ell_mu][n_mu-1]

        if not callable(self.selection) and not callable(self.weight) \
                and ell_mu == ell_nu:
            if n_mu == n_nu:
                shot_noise = rmax**3 * spherical_besselj(ell_mu+1, u_mu)**2 / 2
            else:
                shot_noise = 0.
        else:
            shot_noise = self._access_shot_noise(mu, nu)

        shot_noise *= (1 + 1/contrast) * M_mu_nu / nbar

        return shot_noise

    def two_point_covariance(self, pivot, part='both', diag=False, nbar=None,
                             b_1=None, f_nl=None, tracer_parameter=1.,
                             contrast=np.inf):
        """Compute 2-point signal, shot noise or full covariance matrix for
        given pivot axis for unpacking indices with or without
        scale-dependence modification by local primordial non-Gaussianity.

        Parameters
        ----------
        pivot : {'natural', 'transposed', 'spectral'}
            Pivot axis order for unpacking indices.
        part : {'both', 'signal', 'shotnoise'}, optional
            If ``'both'`` (default), compute the sum of the signal and shot
            noise parts.  If ``'signal'`` or ``'shotnoise'``, compute
            only the corresponding part.
        diag : bool, optional
            If `True` (default is `False`), only compute the diagonal
            entries of the covariance matrix.
        nbar : float or None, optional
            Mean particle number density (in cubic :math:`h`/Mpc).  If
            `part` is ``'shotnoise'`` or ``'both'``, this cannot be `None`
            (default).
        b_1 : float or None, optional
            Scale-independent linear bias of the tracer particles at the
            current epoch.  If `part` is ``'signal'`` or ``'both'``, this
            cannot be `None` (default).
        f_nl : float or None, optional
            Local primordial non-Gaussianity parameter (default is `None`).
        tracer_parameter : float, optional
            Tracer species--dependent parameter for bias modulation
            (default is 1.).
        contrast : float, optional
            Effectively downscale `nbar` by ``1 + 1/constrast``.  Default
            is ``numpy.inf``.

        Returns
        -------
        two_point_covar : complex :class:`numpy.ndarray`
            2-point covariance matrix pivoted at given axis order.

        Raises
        ------
        ValueError
            If `b_1` is `None` when `part` is ``'signal'`` or ``'both'``,
            or `nbar` is `None` when `part` is ``'shotnoise'``
            or ``'both'``.
        ValueError
            If `part` is not a recognised 2-point covariance component.
        ValueError
            If `pivot` is ``'root'`` or ``'scale'``.

        """
        if b_1 is None and part in ['signal', 'both']:
            raise ValueError(
                "`b_1` cannot be None if `part` is 'signal' or 'both'. "
            )
        if nbar is None and part in ['shotnoise', 'both']:
            raise ValueError(
                "`nbar` cannot be None if `part` is 'shotnoise' or 'both'. "
            )
        if pivot in ['root', 'scale']:
            raise ValueError(
                "Pivot by 'root' or 'scale' is not supported by this method. "
                "Apply `variance` method instead for order-collapse cases "
                "where no masking, selection, weighting, evolution or "
                "geometrical effects are present. "
            )

        scale_mod_kwargs = dict(f_nl=f_nl, tracer_parameter=tracer_parameter)
        if part == 'both':
            two_point_component = lambda mu, nu: \
                self.two_point_signal(mu, nu, b_1, **scale_mod_kwargs) \
                + self.two_point_shot_noise(mu, nu, nbar, contrast=contrast)
        elif part == 'signal':
            two_point_component = lambda mu, nu: \
                self.two_point_signal(mu, nu, b_1, **scale_mod_kwargs)
        elif part == 'shotnoise':
            two_point_component = lambda mu, nu: \
                self.two_point_shot_noise(mu, nu, nbar, contrast=contrast)
        else:
            raise ValueError(f"Invalid covariance part: {part}. ")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index_vector = self._natural_indices.unfold(
                pivot, return_only='index'
            )

        dim_covar = len(index_vector)
        two_point_covar = np.zeros((dim_covar, dim_covar), dtype=complex)
        if diag:
            for diag_idx in range(dim_covar):
                mu = index_vector[diag_idx]
                two_point_covar[diag_idx, diag_idx] = two_point_component(
                    mu, mu
                )
        else:
            for row_idx in range(dim_covar):
                for col_idx in range(row_idx+1):
                    mu, nu = index_vector[row_idx], index_vector[col_idx]
                    two_point_covar[row_idx, col_idx] = two_point_component(
                        mu, nu
                    )

            triu_indices = np.triu_indices(dim_covar, k=1)
            two_point_covar[triu_indices] = np.conj(
                two_point_covar.T[triu_indices]
            )

        return two_point_covar

    def mode_variance(self, pivot, part='both', nbar=None, b_1=None,
                      f_nl=None, tracer_parameter=1., contrast=np.inf):
        """Compute the signal, shot noise or total mode variance for given
        pivot axis for unpacking indices, reduced to the simplest case of
        no masking, selection, weighting, evolution or geometrical effects
        and with or without scale-dependence modification by local
        primordial non-Gaussianity.

        Parameters
        ----------
        pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Pivot axis order for unpacking indices.
        part : {'both', 'signal', 'shotnoise'}, optional
            If ``'both'`` (default), compute the sum of the signal and shot
            noise parts.  If ``'signal'`` or ``'shotnoise'``, compute
            only the corresponding part.
        nbar : float or None, optional
            Mean particle number density (in cubic :math:`h`/Mpc).  If
            `part` is ``'shotnoise'`` or ``'both'``, this cannot be `None`
            (default).
        b_1 : float or None, optional
            Scale-independent linear bias of the tracer particles at the
            current epoch.  If `part` is ``'signal'`` or ``'both'``, this
            cannot be `None` (default).
        f_nl : float or None, optional
            Local primordial non-Gaussianity parameter (default is `None`).
        tracer_parameter : float, optional
            Tracer species--dependent parameter for bias modulation
            (default is 1.).
        contrast : float, optional
            Effectively downscale `nbar` by ``1 + 1/constrast``.  Default
            is ``numpy.inf``.

        Returns
        -------
        mode_variance : float :class:`numpy.ndarray`
            2-point mode variance vector pivoted at given axis order.

        Raises
        ------
        ValueError
            If `b_1` is `None` when `part` is ``'signal'`` or
            ``'both'``, or `nbar` is `None` when `part` is ``'shotnoise'``
            or ``'both'``.
        ValueError
            If `part` is not a recognised 2-point covariance component.

        """
        if b_1 is None and part in ['signal', 'both']:
            raise ValueError(
                "`b_1` cannot be None if `part` is 'signal' or 'both'. "
            )
        if nbar is None and part in ['shotnoise', 'both']:
            raise ValueError(
                "`nbar` cannot be None if `part` is 'shotnoise' or 'both'. "
            )

        kappa = self.disc.normalisations
        p_k = self._mode_powers

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index_vector = self._natural_indices.unfold(
                pivot, return_only='index'
            )

        collapse_dof_correction = (pivot in ['scale', 'root'])

        mode_variance = np.zeros(len(index_vector))
        for idx, index in enumerate(index_vector):
            ell, n_idx = index[0], index[-1] - 1

            unnormalised_variance = 0.
            if part in ['signal', 'both']:
                b_k = b_1
                if f_nl is not None:
                    b_k += f_nl * (b_1 - tracer_parameter) \
                        * self._mode_scale_dependence_modifications[ell][n_idx]
                unnormalised_variance += b_k**2 * p_k[ell][n_idx]
            if part in ['shotnoise', 'both']:
                unnormalised_variance += (1 + 1/contrast) / nbar

            mode_variance[idx] = unnormalised_variance / kappa[ell][n_idx]

            if collapse_dof_correction:
                mode_variance[idx] /= 2*ell + 1

        return mode_variance

    @property
    def _mode_powers(self):
        """Power of the underlying matter distribution at the discretised
        Fourier modes.

        Returns
        -------
        dict of {int: float :class:`numpy.ndarray``}
            Mode power unmodulated by tracer bias.

        """
        if self._mode_powers_ is not None:
            return self._mode_powers_

        self._mode_powers_ = {}
        for ell in self.disc.degrees:
            self._mode_powers_[ell] = self.matter_power_spectrum(
                self.disc.wavenumbers[ell]
            )

        return self._mode_powers_

    @property
    def _mode_scale_dependence_modifications(self):
        """Scale-dependent modification to the scale-independent linear
        bias at discretised Fourier modes to be modulated by local
        primordial non-Gaussianity and tracer parameter.

        Returns
        -------
        dict of {int: float :class:`numpy.ndarray`}
            Mode scale-dependent modification.

        Raises
        ------
        ValueError
            If :attr:`cosmo` is `None`.

        """
        if self._mode_scale_dependence_modifications_ is not None:
            return self._mode_scale_dependence_modifications_

        if self.cosmo is None:
            raise ValueError("`cosmo` cannot be None for scale modification. ")

        scale_dependence_modification_kernel = scale_dependence_modification(
            self.cosmo, self.redshift
        )

        self._mode_scale_dependence_modifications_ = {}
        for ell in self.disc.degrees:
            self._mode_scale_dependence_modifications_[ell] = \
                scale_dependence_modification_kernel(
                    self.disc.wavenumbers[ell]
                )

        return self._mode_scale_dependence_modifications_

    @property
    def _fixed_angular_sums(self):
        """Pre-computed angular sums for each fixed pair of coupling
        coefficient indices over the summed angular coupling coefficient
        indices.

        Returns
        -------
        dict
            Angular sums for all pairs of coupling coefficient indices over
            the summed angular coupling coefficient index.

        """
        if self._fixed_angular_sums_ is not None:
            return self._fixed_angular_sums_

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            operatable_index_vector = map(
                lambda tup: (tup[0], tup[1], None),
                self._natural_indices.unfold('natural', return_only='index')
            )
        index_vector = sorted(list(set(operatable_index_vector)))

        angular_couplings = self.couplings['angular']

        def _angular_sum(_ind_pair):

            partial_mu, partial_nu = _ind_pair

            _sum = {}
            for ell in self.disc.degrees:
                M_mu_ = angular_couplings[partial_mu][ell]
                M_nu_ = angular_couplings[partial_nu][ell]
                _sum.update({ell: np.sum(M_mu_ * np.conj(M_nu_))})

            return _sum

        index_pair_vector = list(product(*(index_vector,)*2))
        index_pair_ang_sum = mpi_compute(
            index_pair_vector, _angular_sum, self.comm,
            logger=self._logger, process_name="fixed angular sum"
        )

        self._fixed_angular_sums_ = defaultdict(dict)
        for ell in self.disc.degrees:
            self._fixed_angular_sums_[ell].update(
                {
                    index_pair : ang_sum[ell]
                    for index_pair, ang_sum
                    in zip(index_pair_vector, index_pair_ang_sum)
                }
            )

        return self._fixed_angular_sums_

    @property
    def _fixed_shot_noise(self):
        """Pre-computed shot noise integral for each fixed pair of coupling
        coefficient indices.

        Returns
        -------
        dict
            Shot noise integral for all pairs of coupling coefficient
            indices.

        """
        if self._fixed_shot_noise_ is not None:
            return self._fixed_shot_noise_

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            operatable_index_vector = map(
                lambda tup: (tup[0], None, tup[2]),
                self._natural_indices.unfold('natural', return_only='index')
            )
        index_vector = sorted(list(set(operatable_index_vector)))

        kwargs = dict(selection=self.selection, weight=self.weight)

        self._fixed_shot_noise_ = {}
        with warnings.catch_warnings(record=True) as any_warnings:
            for mu, nu in product(*(index_vector,)*2):
                k_mu = self.disc.wavenumbers[mu[0]][mu[-1]-1]
                k_nu = self.disc.wavenumbers[nu[0]][nu[-1]-1]
                self._fixed_shot_noise_[(mu, nu)] = rad_int(
                    lambda r: \
                        _shot_noise_kernel(r, mu, nu, k_mu, k_nu, **kwargs),
                    self.disc.attrs['boundary_radius']
                )

        unique_warning_msgs = set(map(
            lambda warning_obj: warning_obj.message, any_warnings
        ))
        for msg in unique_warning_msgs:
            warnings.warn(
                "Shot noise integration warning emitted: {}.\n".format(msg),
                category=IntegrationWarning
            )

        return self._fixed_shot_noise_

    def _access_couplings(self, coupling_type, mu):

        if coupling_type == 'angular':
            _tuple_key = (mu[0], mu[1], None)
        else:
            _tuple_key = (mu[0], None, mu[2])
        return self.couplings[coupling_type][_tuple_key]

    def _access_angular_sums(self, ell, mu, nu):

        _tuple_key = ((mu[0], mu[1], None), (nu[0], nu[1], None))

        return self._fixed_angular_sums[ell][_tuple_key]

    def _access_shot_noise(self, mu, nu):

        _tuple_key = ((mu[0], None, mu[2]), (nu[0], None, nu[2]))

        return self._fixed_shot_noise[_tuple_key]
