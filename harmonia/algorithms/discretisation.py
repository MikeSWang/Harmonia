"""
Discrete Fourier spectrum (:mod:`~harmonia.algorithms.discretisation`)
===========================================================================

Discretise Fourier spectrum of cosmological fields.

.. autosummary::

    DiscreteSpectrum

|

"""
import logging

import numpy as np

from .bases import spherical_besselj, spherical_besselj_root


class DiscreteSpectrum:
    r"""Discrete Fourier spectrum for given radial boundary conditions,
    indexed by spherical degree :math:`\ell` associated with spherical
    harmonic and Bessel functions.

    When a boundary condition is prescribed at some maximum radius
    :math:`r = R`, the allowed wave numbers for the discretised spectrum
    are

    .. math::

        k_{\ell n} = \frac{u_{\ell n}}{R}, \quad \text{where} \quad
        \{ u_{\ell n}: n = 1, \dots, n_{\mathrm{max},\ell} \}_{\ell}

    are roots of the spherical Bessel functions of order :math:`\ell` if
    the boundary condition is Dirichlet, or their derivatives if the
    boundary condition is Neumann.  The spherical depths :math:`\{
    n_{\mathrm{max},\ell} \}` are the maximal number of radial wave numbers
    allowed in the specified range, and are indexed by :math:`(\ell, n)`
    doublet tuples.  The normalisation coefficients derived from
    completeness relations are

    .. math::

        \kappa_{\ell n} = \frac{2}{R^3} j_{\ell+1}^{-2}(u_{\ell n})

    for Dirichlet boundary conditions, and

    .. math::

        \kappa_{\ell n} = \frac{2}{R^3} j_{\ell}^{-2}(u_{\ell n})
        \Bigg[ 1 - \frac{\ell(\ell+1)}{u_{\ell n}^2} \Bigg]^{-1}

    for Neumann boundary conditions.

    Parameters
    ----------
    radius : float
        Boundary radius.
    condition : {'dirichlet', 'neumann'}
        Either Dirichlet or Neumann boundary condition.
    cutoff : float
        Fourier scale upper cutoff.
    maxdeg : int or None, optional
        Maximum spherical degree (default is `None`).
    cuton : float, optional
        Fourier scale lower cutoff (default is 0.).
    mindeg : int, optional
        Minimum spherical degree (default is 0).

    Attributes
    ----------
    degrees : int, array_like
        Spherical degrees associated with the discrete spectrum.
    depths : int, array_like
        Spectral depths associated with the discrete spectrum.
    roots : :obj:`dict` of {int: :class:`numpy.ndarray`}
        Spherical Bessel roots associated with the discrete spectrum.
    mode_count : int
        Total number of allowed spectral modes, counting spherical order
        multuplicities.
    attrs : dict
        Discrete spectrum information, which contains the following keys:
        ``'min_wavenumber'``, ``'max_wavenumber'`` for minimum and maximum
        wave numbers; ``'boundary_radius'``, ``'bounded_volume'`` for the
        bounding radius and volume; ``'boundary_condition'`` for the
        boundary condition type.
    wavenumbers
    root_indices
    normalisations

    """

    _logger = logging.getLogger("DiscreteSpectrum")

    def __init__(self, radius, condition, cutoff, maxdeg=None, cuton=0.,
                 mindeg=0):

        condition = self._alias(condition)

        self.degrees, self.depths, self.roots, self.mode_count = \
            self.discretise(radius, condition, cuton, cutoff, mindeg, maxdeg)

        self.attrs = {
            'min_wavenumber': cuton,
            'max_wavenumber': cutoff,
            'boundary_radius': radius,
            'bounded_volume': (4*np.pi/3) * radius**3,
            'boundary_condition': condition,
        }

        self._wavenumbers = None
        self._root_indices = None
        self._normalisations = None

        self._logger.info(
            "%s computed: %d degrees and %d modes in total. ",
            self.__str__(),
            len(self.degrees),
            self.mode_count,
        )

    def __str__(self):
        return "Spectrum({0}, radius={1}, {2} <= wavenumber <= {3})".format(
            self.attrs['boundary_condition'],
            self.attrs['boundary_radius'],
            self.attrs['min_wavenumber'],
            self.attrs['max_wavenumber'],
        )

    @property
    def wavenumbers(self):
        r"""Discrete wavenumbers :math:`k_{\ell n}`.

        Returns
        -------
        :obj:`dict` of {int: :class:`numpy.ndarray`}
            Wavenumbers.

        """
        if self._wavenumbers is not None:
            return self._wavenumbers

        self._wavenumbers = {
            ell : self.roots[ell] / self.attrs['boundary_radius']
            for ell in self.degrees
        }
        self._logger.info("Spectral wavenumbers computed. ")

        return self._wavenumbers

    @property
    def root_indices(self):
        r"""Root indices :math:`(\ell, n)`.

        Returns
        -------
        :obj:`dict` of {int: (int, int)}
            Root indices.

        """
        if self._root_indices is not None:
            return self._root_indices

        self._root_indices = {
            ell: [(ell, n) for n in range(1, nmax+1)]
            for ell, nmax in zip(self.degrees, self.depths)
        }
        self._logger.info(
            "Doublet indices compiled for spectral wavenumbers. ",
        )

        return self._root_indices

    @property
    def normalisations(self):
        r"""Normalisation coefficients :math:`\kappa_{\ell n}`.

        Returns
        -------
        :obj:`dict` of {int: :class:`numpy.ndarray`}
            Normalisation coefficients.

        """
        if self._normalisations is not None:
            return self._normalisations

        radius = self.attrs['boundary_radius']
        condition = self.attrs['boundary_condition']

        if condition == 'dirichlet':
            self._normalisations = {
                ell: 2 / radius**3 \
                    / spherical_besselj(ell+1, self.roots[ell])**2
                for ell in self.degrees
            }
        elif condition == 'neumann':
            self._normalisations = {
                ell: 2 / radius**3 \
                    / spherical_besselj(ell, self.roots[ell])**2 \
                    / (1 - ell * (ell+1) / np.square(self.roots[ell]))
                for ell in self.degrees
            }
        self._logger.info(
            "Normalisations computed for spectral wavenumbers. ",
        )

        return self._normalisations

    @staticmethod
    def discretise(radius, condition, kmin, kmax, ellmin, ellmax):
        """
        Parameters
        ----------
        radius : float
            Boundary radius.
        condition : {'dirichlet', 'neumann'}
            Either Dirichlet or Neumann boundary condition.
        kmin, kmax : float
            Minimum and maximum wavenumbers.
        ellmin, ellmax : int or None
            Minimum and maximum spherical degrees.

        Returns
        -------
        degrees : int, array_like
            Spherical function degrees.
        depths : int, array_like
            Maximal radial numbers, i.e. the number of allowed radial modes
            for each degree.
        roots : :obj:`dict` of float :class:`numpy.ndarray`
            Spherical Bessel roots.
        mode_count : int
            Total number of spectral modes, counting spherical order
            multiplicities.

        Raises
        ------
        ValueError
            If `condition` does not correpond to either 'Dirichlet' or
            'Neumann'.

        """
        _logger_ = logging.getLogger('discretisation')

        if condition.lower().startswith('d'):
            derivative = False
        elif condition.lower().startswith('n'):
            derivative = True
        else:
            raise ValueError(f"Invalid boundary `condition`: {condition}. ")

        degrees, depths = [], []
        roots = {}
        ell, mode_count = ellmin, 0
        while True:
            if ellmax is not None:
                if ell > ellmax:
                    _logger_.debug("Maximum degree reached. ")
                    break

            u_ell, n_ell = [], 0
            current_root = spherical_besselj_root(
                ell,
                n_ell + 1,
                derivative=derivative,
            )
            while kmin * radius <= current_root <= kmax * radius:
                u_ell.append(current_root)
                n_ell += 1
                current_root = spherical_besselj_root(
                    ell,
                    n_ell + 1,
                    derivative=derivative,
                )

            if n_ell == 0:
                _logger_.info("No more roots found. Last degree is %d. ", ell)
                break
            else:
                roots[ell] = np.array(u_ell)
                degrees.append(ell)
                depths.append(n_ell)
                mode_count += (2*ell + 1) * n_ell
                _logger_.debug("Results for degree %d appended. ", ell)

                ell += 1
                _logger_.debug("Moving on to next degree: %d. ", ell)

        return degrees, depths, roots, mode_count

    @staticmethod
    def _alias(condition):

        if condition.lower().startswith('d'):
            alias = 'dirichlet'
        elif condition.lower().startswith('n'):
            alias = 'neumann'
        else:
            raise ValueError(f"Invalid boundary `condition`: {condition}. ")

        return alias
