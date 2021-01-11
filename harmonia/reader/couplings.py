r"""
Spherical couplings (:mod:`~harmonia.reader.couplings`)
===========================================================================

Compute spherical Fourier couplings coefficients for given survey
spefications and cosmological models.

.. autosummary::

    SphericalCoefficientWarning
    Couplings


Kernels
---------------------------------------------------------------------------

Coupling kernels are integrands without the coordinate Jacobian, which may
include the following factors:

    * angular mask :math:`M(\hat{\mathbf{r}})`;
    * radial selection :math:`\phi(r)`;
    * radial weight :math:`w(r)` and its derivative :math:`w'(r)`;
    * clustering evolution, :math:`D(z)`, which is the linear growth
      factor normalised to unity at the :math:`z = 0` epoch;
    * linear bias + clustering evolution
      :math:`G(z, k) = b(z, k) D(z) / b(z_*, k) D(z_*)` normalised to unity
      at a fiducial epoch :math:`z_*`, where :math:`b(z, k)` is the
      scale-dependent linear bias;
    * linear growth rate + clustering evolution
      :math:`F(z) = f(z) D(z) / f(z_*) D(z_*)` normalised to unity at a
      fiducial epoch :math:`z_*`, where :math:`f(z)` is the linear
      growth rate;
    * differential Alcock--Paczynski (AP) distortion

      .. math::

          \gamma(z) = \frac{
              \operatorname{d}\!\breve{\chi}(z)
          }{
              \operatorname{d}\!\chi(z)
          } \,,

      where :math:`\breve{r} = \breve{\chi}(z)` is the distance converted
      from redshift in a fiducial cosmology rather than from the true
      comoving distance--redshift correspondence :math:`z = \chi^{-1}(r)`.


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
           \int \operatorname{d}\!r \, r^2 w(\breve{r})
           j_{\ell_\mu}(k_{\ell_\mu n_\mu} \breve{r})
           j_{\ell_\nu}(k_{\ell_\nu n_\nu} r)
           G(z(r), k_{\ell_\nu n_\nu}) \phi(r) \,, \\
       \Upsilon_{\mu\nu} &=
           \frac{\kappa_{\ell_\nu n_\nu}}{k_{\ell_\nu n_\nu}}
           \int \operatorname{d}\!r \, r^2
           \frac{\operatorname{d}\!}{\operatorname{d}\!\breve{r}}
           \left[ w(\breve{r}) j_{\ell_\mu}(k_{\ell_\mu n_\mu} \breve{r})
               \right]
           j'_{\ell_\nu}(k_{\ell_\nu n_\nu} r)
           \gamma(z(r)) F(z(r)) \phi(r) \,,
   \end{align*}

over the spherical volume element, where :math:`k_{\ell n}` is the
discrete wavenumber.

When there is no angular masking (i.e. :math:`M(\hat{\mathbf{r}})` is
constant), the coupling coefficients reduce to :math:`M_{\mu\nu} =
\delta_{\mu\nu}`; if in addition radial selection, weighting and
evolutionary effects are all absent and there is no AP correction, then
:math:`M_{\mu\nu} \Phi_{\mu\nu} = \delta_{\mu\nu}`.

|

"""
import logging
import warnings

import numpy as np

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.integration import (
    angular_integral,
    pixelated_angular_integral,
    radial_integral,
)
from harmonia.utils import mpi_compute, restore_warnings
from ._kernels import angular_kernel, radial_kernel, RSD_kernel


class SphericalCoefficientWarning(UserWarning):
    """Warning issued for poorly determined spherical coefficient.

    """


class Couplings:
    """Angular, radial and RSD coupling coefficients computed for given
    survey and cosmological specifications.

    Notes
    -----
    Survey specification functions must be given in spherical coordinates
    and may include the following:

        * 'mask' for angular mask;
        * 'selection', 'weight' and 'weight_derivative' for radial
          selection, weighting and weight derivative.

    Cosmological specification functions must be given in redshift
    and/or the radial coordinate and may include the following:

        * 'z_from_r' and 'chi_of_z' for cosmological comoving
          distance-to-redshift conversion and fiducial
          redshift-to-comoving distance conversion;
        * 'bias_evolution', 'growth_evolution' for linear bias and
          linear growth rate, normalised to unity at a fiducial epoch;
        * 'clustering_evolution' for clustering evolution as a
          function of the redshift normalised to unity at the
          current epoch;
        * 'differential_AP_distortion' for differential AP distortion
          as a function of the redshift.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    survey_specs : dict{str: callable or None} or None, optional
        Survey specification functions as detailed above.
    cosmo_specs : dict{str: callable or None} or None, optional
        Cosmological specification functions as detailed above.
    initialise : bool, optional
        If `True`, compile all coupling coefficients upon creation.
    external_angular_couplings : dict{tuple(tuple, tuple): complex} or None, optional
        Pre-compute angular couplings (default is `None`).
    pixelate : int or None, optional
        If not `None` (default), this sets the 'NSIDE' parameter of
        `healpy` pixelation for angular coupling integration.
    comm : :class:`mpi4py.MPI.Comm` *or None, optional*
        MPI communicator (default is `None`).

    Attributes
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the couplings.
    couplings : dict{str: dict}
        Directory for all coupling coefficients of different
        coupling types.

    """

    _coupling_types = ['angular', 'radial', 'rsd']
    _survey_specs = dict.fromkeys([
        'mask', 'selection', 'weight', 'weight_derivative'
    ])

    _cosmo_specs = dict.fromkeys([
        'z_from_r', 'chi_of_z',
        'bias_evolution', 'growth_evolution',
        'clustering_evolution',
        'differential_AP_distortion',
    ])

    def __init__(self, disc, survey_specs=None, cosmo_specs=None,
                 initialise=True, external_angular_couplings=None,
                 pixelate=None, comm=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm
        self.initialised = False

        self.disc = disc

        if survey_specs is not None:
            self._survey_specs.update(survey_specs)
        if cosmo_specs is not None:
            self._cosmo_specs.update(cosmo_specs)

        self.couplings = {
            coupling_type: {}
            for coupling_type in self._coupling_types
        }
        if external_angular_couplings is not None:
            self.load_angular_couplings(external_angular_couplings)
        if initialise:
            self.compile_couplings(pixelate=pixelate)

    def __getstate__(self):

        state = {
            'disc': self.disc.__getstate__(),
            'couplings': self.couplings,
        }

        return state

    def __setstate__(self, state):

        for attr, value in state.items():
            if attr == 'disc':
                self.disc = DiscreteSpectrum._from_state(state['disc'])
            else:
                # NOTE: Strip '_' for backward compatibility; may be
                # removed in the future.
                setattr(self, attr.strip('_'), value)

    def __getitem__(self, key):
        r"""Access coupling coefficient by key.

        Notes
        -----
        Only accessible if initialised with :meth:`compile_couplings`.
        The access key is a tuple specifying the coupling type and the
        pair of triplet/double indices, e.g.
        ``['angular', (0, 0), (3, -1, 1)]`` for :math:`M_{0,0,3,-1}`.

        Parameters
        ----------
        key : tuple
            Coefficient access key.  Must contain elements of types
            specified above.

        Returns
        -------
        float or complex
            Coupling coefficient.

        Raises
        ------
        AttributeError
            If the coupling coefficients have not been initialised.

        """
        if not self.initialised:
            raise AttributeError(
                "Unitialised state: coupling coefficients not compiled. "
                "Please call `compile_couplings` first."
            )

        coupling_type, mu, nu = key

        if coupling_type == 'angular':
            mu, nu = (mu[0], mu[1]), (nu[0], nu[1])
            # Use parity to return an angular coupling coefficient
            # that is not internally stored.
            if mu > nu:
                return np.conj(self.couplings['angular'][nu, mu])

            return self.couplings['angular'][mu, nu]

        mu, nu = (mu[0], mu[-1]), (nu[0], nu[-1])

        return self.couplings[coupling_type][mu, nu]

    def load_angular_couplings(self, angular_couplings):
        """Load pre-computed angular coupling coefficients which are
        independent of the cosmological model.

        Parameters
        ----------
        angular_couplings : dict{tuple(tuple, tuple): complex}
            Pre-compute angular couplings.

        Raises
        ------
        ValueError
            If the number of entries in `angular_couplings` do not match
            the class instance.

        """
        num_unordered_index_pair = sum(range(
            1, sum([2 * deg + 1 for deg in self.disc.degrees])
        ))
        num_ordered_index_pair = sum(
            [2 * deg + 1 for deg in self.disc.degrees]
        ) ** 2

        if len(angular_couplings) \
                not in [num_unordered_index_pair, num_ordered_index_pair]:
            raise ValueError(
                "Number of index pairs in loaded angular couplings "
                "do not match the `disc` attribute."
            )

        self.couplings.update({'angular': angular_couplings})

    def compile_couplings(self, pixelate=None):
        """Compile all coupling coefficients.

        Parameters
        ----------
        pixelate : int or None, optional
            If not `None` (default), this sets the 'NSIDE' parameter of
            `healpy` pixelation for angular coupling integration.

        """
        for coupling_type in self._coupling_types:
            # Only compile empty coupling directories.
            if not self.couplings[coupling_type]:
                self._compile_couplings_by_type(
                    coupling_type, pixelate=pixelate
                )

        self.initialised = True

    def compute_coefficient(self, mu, nu, coupling_type, pixelate=None):
        r"""Compute coupling coefficients for given triplet indices.

        Parameters
        ----------
        mu, nu : tuple(int, int) or tuple(int, int, int)
            Coefficient triplet or reduced doublet index.
        coupling_type : {'angular', 'radial', 'RSD'}
            Coupling type.
        pixelate : int or None, optional
            If not `None` (default), this sets the 'NSIDE' parameter of
            `healpy` pixelation for computing angular coupling
            coefficients.

        Returns
        -------
        cofficient : float or complex :class:`numpy.ndarray`
            Coupling coefficient of the specified type.

        """
        if coupling_type == 'angular':
            if not callable(self._survey_specs['mask']):  # Kronecker delta
                coefficient = complex(mu[0] == nu[0] and mu[1] == nu[1])
            else:
                if pixelate:
                    coefficient = pixelated_angular_integral(
                        lambda theta, phi: angular_kernel(
                            theta, phi, mu, nu,
                            mask=self._survey_specs['mask']
                        ), nside=pixelate
                    )
                else:
                    with warnings.catch_warnings(record=True) as any_warning:
                        coefficient = angular_integral(
                            lambda theta, phi: angular_kernel(
                                theta, phi, mu, nu,
                                mask=self._survey_specs['mask']
                            )
                        )
                    if any_warning and not np.isclose(coefficient, 0.):
                        warnings.warn(
                            "Poorly determined angular coupling coefficients "
                            "for index pair {} and {}.".format(mu, nu),
                            category=SphericalCoefficientWarning
                        )
            return coefficient

        k_mu = self.disc.wavenumbers[mu[0], mu[-1]]
        k_nu = self.disc.wavenumbers[nu[0], nu[-1]]
        kappa_nu = self.disc.normalisations[nu[0], nu[-1]]

        spec_funcs = {}
        checklists = {
            '_survey_specs': ['selection', 'weight'],
            '_cosmo_specs': ['z_from_r', 'chi_of_z', 'clustering_evolution'],
        }

        if coupling_type == 'radial':
            checklists['_cosmo_specs'].extend(['bias_evolution'])
            spec_funcs.update({
                spec: getattr(self, spec_type)[spec]
                for spec_type, spec_list in checklists.items()
                for spec in spec_list
            })

            if (mu[0] == nu[0]) and not any(map(callable, spec_funcs.items())):
                coefficient = float(mu[-1] == nu[-1])  # Kronecker delta
            else:
                with warnings.catch_warnings(record=True) as any_warning:
                    coefficient = kappa_nu * radial_integral(
                        lambda r: radial_kernel(
                            r, mu, nu, k_mu, k_nu, **spec_funcs
                        ),
                        self.disc.attrs['boundary_radius']
                    )
                if any_warning:
                    warnings.warn(
                        "Poorly determined radial coupling coefficients "
                        "for index pair {} and {}.".format(mu, nu),
                        category=SphericalCoefficientWarning
                    )
            return coefficient

        if coupling_type == 'rsd':
            checklists['_survey_specs'].extend(['weight_derivative'])
            checklists['_cosmo_specs'].extend(
                ['growth_evolution', 'differential_AP_distortion']
            )
            spec_funcs.update({
                spec: getattr(self, spec_type)[spec]
                for spec_type, spec_list in checklists.items()
                for spec in spec_list
            })

            with warnings.catch_warnings(record=True) as any_warning:
                coefficient = kappa_nu / k_nu * radial_integral(
                    lambda r: RSD_kernel(r, mu, nu, k_mu, k_nu, **spec_funcs),
                    self.disc.attrs['boundary_radius']
                )
            if any_warning:
                warnings.warn(
                    "Poorly determined RSD coupling coefficients "
                    "for index pair {} and {}.".format(mu, nu),
                    category=SphericalCoefficientWarning
                )

            return coefficient

        raise ValueError(f"Unknown coupling type: {coupling_type}.")

    def save(self, output_file):
        """Save compiled couplings as a .npz file.

        Parameters
        ----------
        output_file : *str or* :class:`pathlib.Path`
            Output file path.

        """
        np.savez(output_file, **self.__getstate__())

    @classmethod
    def load(cls, input_file, comm=None):
        """Load compiled couplings from a .npz file.

        Parameters
        ----------
        input_file : *str or* :class:`pathlib.Path`
            Input file path.

        """
        state_data = np.load(input_file, allow_pickle=True)
        state = {}
        for attr in state_data.files:
            try:
                state.update({attr: state_data[attr].item()})
            except ValueError:
                state.update({attr: state_data[attr]})

        self = object.__new__(cls)

        self.__setstate__(state)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm

        self.initialised = True

        return self

    def _compile_couplings_by_type(self, coupling_type, pixelate=None):

        indices_to_compile = self._gen_operable_indices(coupling_type)
        index_compiler = lambda index: self._compile_couplings_by_index(
            index, coupling_type, pixelate=pixelate
        )

        if self.comm is None or self.comm.rank == 0:
            self.logger.info(
                "Compiling %s couplings...", self._alias(coupling_type)
            )
        if self.comm is not None:
            self.comm.Barrier()

        with warnings.catch_warnings(record=True) as any_warnings:
            list_of_compiled_couplings_ = mpi_compute(
                indices_to_compile, index_compiler, comm=self.comm,
                process_name="{} couplings compilation".format(
                    self._alias(coupling_type)
                )
            )

        if any_warnings:
            restore_warnings(any_warnings)
            warnings.warn(
                "Poor coefficient determinations for {} couplings."
                .format(coupling_type),
                SphericalCoefficientWarning
            )

        for compiled_couplings in list_of_compiled_couplings_:
            self.couplings[coupling_type].update(compiled_couplings)

        if self.comm is not None:
            self.comm.Barrier()
        if self.comm is None or self.comm.rank == 0:
            self.logger.info(
                "... compiled %s couplings.", self._alias(coupling_type)
            )

    def _compile_couplings_by_index(self, fixed_index, coupling_type,
                                    pixelate=None):

        mu = (fixed_index[0], fixed_index[1]) if coupling_type == 'angular' \
            else (fixed_index[0], fixed_index[-1])

        # Reduce the number of indices to compile by exploting
        # exchange symmetry between indices.
        above_from = mu if coupling_type == 'angular' else None
        variable_indices = self._gen_operable_indices(
            coupling_type, above_from=above_from
        )

        couplings_for_index = {}
        for nu in variable_indices:
            couplings_for_index.update({
                (mu, nu): self.compute_coefficient(
                    mu, nu, coupling_type, pixelate=pixelate
                )
            })

        return couplings_for_index

    def _gen_operable_indices(self, coupling_type, above_from=None):

        if coupling_type == 'angular':
            operable_indices = [
                (ell, m)
                for ell in self.disc.degrees
                for m in range(- ell, ell + 1)
            ]
            if above_from is not None:
                operable_indices = [
                    index for index in operable_indices if index >= above_from
                ]
        else:
            operable_indices = [
                (ell, n)
                for ell, nmax in zip(self.disc.degrees, self.disc.depths)
                for n in range(1, nmax + 1)
            ]

        return operable_indices

    @staticmethod
    def _alias(coupling_type):

        return 'RSD' if coupling_type == 'rsd' else coupling_type


def _group_couplings(couplings):
    """Change a directory of couplings from nested dictionaries to a
    dictionary of arrays for mass access to raidial and RSD coupling
    coefficients of adjacent indices.

    Parameters
    ----------
    couplings : :class:`~.reader.couplings.Couplings`
        Couplings object (default is `None`).

    Returns
    -------
    grouped_couplings : dict{tuple: list}
        Radial and RSD couplings grouped for mass access.

    """
    disc = couplings.disc

    # pylint: disable=protected-access
    key_indices = couplings._gen_operable_indices('radial')

    grouped_couplings = {}
    for coupling_type in ['radial', 'rsd']:
        coupling_arrays_dir = {
            key_index: [
                np.asarray([
                    couplings[coupling_type, key_index, (ell, n)]
                    for n in range(1, nmax + 1)
                ])
                for ell, nmax in zip(disc.degrees, disc.depths)
            ]
            for key_index in key_indices
        }
        grouped_couplings.update({coupling_type: coupling_arrays_dir})

    return grouped_couplings
