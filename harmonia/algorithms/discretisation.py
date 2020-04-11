"""
Spectrum discretisation (:mod:`~harmonia.algorithms.discretisation`)
===========================================================================

Discretise the Fourier spectrum under boundary conditions.

.. autosummary::

    DiscreteSpectrum

|

"""
import logging

import numpy as np

from .bases import spherical_besselj, spherical_besselj_root


class DiscreteSpectrum:
    r"""Discrete Fourier spectrum under radial boundary conditions.

    The spectral modes are indexed by tuple :math:`(\ell, n)`, where
    :math:`\ell` is the spherical degree associated with the spherical
    harmonic and Bessel functions, and :math:`n` is the spherical depth
    associated with roots of the spherical Bessel function.

    The discrete wavenumbers are

    .. math::

        k_{\ell n} = \frac{u_{\ell n}}{R} \,,
        \quad \text{where} \quad
        \{ u_{\ell n}: n = 1, \dots, n_{\textrm{max},\ell} \}_{\ell}

    are zeros of the spherical Bessel functions of order :math:`\ell` if
    the boundary condition is Dirichlet, or zeros of their derivatives if
    the boundary condition is Neumann.  The maximum spherical depth
    :math:`n_{\textrm{max},\ell}` corresponds to the largest wavenumber
    allowed in the cutoff range.

    The normalisation coefficients derived from completeness relations are

    .. math::

        \kappa_{\ell n} =
            \begin{cases}
                (2/R^3) j_{\ell+1}^{-2}(u_{\ell n}) \,,
                \quad \text{for Dirichlet boundary conditions;} \\
                (2/R^3) j_{\ell}^{-2}(u_{\ell n}) \left[
                    1 - \ell (\ell + 1) / u_{\ell n}^2
                \right]^{-1} \,,
                \quad \text{for Neumann boundary conditions.}
            \end{cases}

    Parameters
    ----------
    radius : float
        Boundary radius.
    condition : {'dirichlet', 'neumann'}
        Either Dirichlet or Neumann boundary condition.
    highcut : float
        Fourier spectrum upper cutoff.
    maxdeg : int or None, optional
        Maximum spherical degree (default is `None`).
    lowcut : float, optional
        Fourier spectrum lower cutoff (default is 0.).
    mindeg : int, optional
        Minimum spherical degree (default is 0).
    comm : :class:`mpi4py.MPI.Comm`
        MPI communicator.

    Attributes
    ----------
    degrees : list of int
        Spherical degrees associated with the discrete spectrum.
    depths : list of int
        Spherical depths associated with each spherical degree.
    mode_counts : list of int
        Total number of spectral modes associated with each spherical
        degree counting spherical order multiplicities.
    roots : dict
        Spherical Bessel roots associated with the discrete spectrum as
        a dictionary accessed by integer spherical degrees.
    attrs : dict
        Discrete spectrum attributes including the following keys:
        'min_wavenumber' and 'max_wavenumber' for minimum and
        maximum wavenumbers, 'boundary_radius' and 'bounded_volume'
        for the boundary radius and bounded volume, and
        'boundary_condition' for the imposed boundary condition.

    """

    def __init__(self, radius, condition, highcut, maxdeg=None, lowcut=0.,
                 mindeg=0, comm=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm

        condition = self._alias(condition)

        self.attrs = {
            'min_wavenumber': lowcut,
            'max_wavenumber': highcut,
            'boundary_radius': radius,
            'bounded_volume': 4./3. * np.pi * radius ** 3,
            'boundary_condition': condition,
        }

        self._discretise(radius, condition, lowcut, highcut, mindeg, maxdeg)

        self._wavenumbers = None
        self._normalisations = None

    def __str__(self):

        str_info = "{}, boundary={}, {}<=wavenumber<={}".format(
            self.attrs['boundary_condition'],
            self.attrs['boundary_radius'],
            np.around(self.attrs['min_wavenumber'], decimals=4),
            np.around(self.attrs['max_wavenumber'], decimals=4),
        )

        return f"{self.__class__.__name__}({str_info})"

    def __setstate__(self, state):

        for attr, value in state.items():
            if attr in ['wavenumbers', 'normalisations']:
                setattr(self, '_' + attr, value)
            else:
                setattr(self, attr, value)

    def __getstate__(self):

        state = {
            attr: getattr(self, attr)
            for attr in [
                'attrs',
                'degrees', 'depths', 'mode_counts',
                'roots', 'wavenumbers', 'normalisations'
            ]
        }

        return state

    @classmethod
    def _from_state(cls, state, comm=None):  # internal classmethod

        self = object.__new__(cls)

        self.__setstate__(state)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.comm = comm

        return self

    @property
    def wavenumbers(self):
        r"""Discrete mode wavenumbers :math:`k_{\ell n}`.

        Returns
        -------
        dict
            Mode wavenumbers as a dictionary accessed by
            doublet tuples :math:`(\ell, n)`.

        """
        if self._wavenumbers is not None:
            return self._wavenumbers

        self._wavenumbers = {
            (ell, n_idx + 1) : u / self.attrs['boundary_radius']
            for ell in self.degrees
            for n_idx, u in enumerate(self.roots[ell])
        }

        if self.comm is None or self.comm.rank == 0:
            self.logger.debug("Spectral mode wavenumbers computed.")

        return self._wavenumbers

    @property
    def normalisations(self):
        r"""Normalisation coefficients :math:`\kappa_{\ell n}`.

        Returns
        -------
        dict
            Normalisation coefficients as a dictionary accessed by
            doublet tuples :math:`(\ell, n)`.

        """
        if self._normalisations is not None:
            return self._normalisations

        if self.attrs['boundary_condition'] == 'dirichlet':
            self._normalisations = {
                (ell, n_idx + 1): 2. \
                    / self.attrs['boundary_radius'] ** 3 \
                    / spherical_besselj(ell + 1, u) ** 2
                for ell in self.degrees
                for n_idx, u in enumerate(self.roots[ell])
            }
        if self.attrs['boundary_condition'] == 'neumann':
            self._normalisations = {
                (ell, n_idx + 1): 2. \
                    / self.attrs['boundary_radius'] ** 3 \
                    / spherical_besselj(ell, u) ** 2 \
                    / (1 - ell * (ell + 1) / u**2)
                for ell in self.degrees
                for n_idx, u in enumerate(self.roots[ell])
            }

        if self.comm is None or self.comm.rank == 0:
            self.logger.debug("Spectral normalisations computed.")

        return self._normalisations

    def _discretise(self, radius, condition, kmin, kmax, ellmin, ellmax):

        # Avoid duplicate logging entries.
        log = self.comm is None or self.comm.rank == 0

        # Initiate containers.
        self.degrees, self.depths, self.mode_counts = [], [], []
        self.roots = {}

        current_ell = ellmin
        while True:
            # Stop if maximum degree specified is reached.
            if ellmax is not None and current_ell > ellmax:
                if log:
                    self.logger.debug("Specfied maximum degree reached.")
                break

            # Initiate the container and counter for the roots and depth of
            # the current degree.
            u_ell, n_ell = [], 0

            # Iterate through spherical Bessel roots.
            current_root = spherical_besselj_root(
                current_ell, n_ell + 1, derivative=(condition == 'neumann')
            )
            while kmin * radius <= current_root <= kmax * radius:
                u_ell.append(current_root)
                n_ell += 1
                current_root = spherical_besselj_root(
                    current_ell, n_ell + 1, derivative=(condition == 'neumann')
                )

            # If iteration did not proceed, end the entire process.
            if n_ell == 0:
                if log:
                    self.logger.debug(
                        "The last degree is %d.", current_ell - 1
                    )
                break

            self.degrees.append(current_ell)
            self.depths.append(n_ell)
            self.mode_counts.append((2 * current_ell + 1) * n_ell)
            self.roots[current_ell] = np.asarray(u_ell)

            if log:
                self.logger.debug("Degree %d included.", current_ell)

            current_ell += 1

        if log:
            self.logger.info(
                "%s computed: %d degrees and %d modes in total.",
                self, len(self.degrees), sum(self.mode_counts)
            )

    @staticmethod
    def _alias(condition):

        if condition.lower().startswith('d'):
            return 'dirichlet'

        if condition.lower().startswith('n'):
            return 'neumann'

        raise ValueError(f"Unknown boundary `condition`: {condition}.")
