"""
Spectrum discretisation (:mod:`~harmonia.algorithms.discretisation`)
===============================================================================

Discretise Fourier spectrum of cosmological fields.

.. autosummary::

    DiscreteSpectrum

|

"""
import logging

import numpy as np

from .bases import spherical_besselj, spherical_besselj_root


class DiscreteSpectrum:
    r"""Discrete Fourier spectrum for given radial boundary conditions.

    When a boundary condition is prescribed at some maximum radius
    :math:`r = R`, the allowed wave numbers for the discretised spectrum are

    .. math::

        k_{\ell n} = \frac{u_{\ell n}}{R}, \quad \text{where} \quad
        \{ u_{\ell n}: n = 1, \dots, n_{\mathrm{max},\ell} \}_{\ell}

    are roots of the spherical Bessel functions of degree :math:`\ell` if the
    boundary condition is Dirichlet, or their derivatives if the boundary
    condition is Neumann.  The spherical depths :math:`\{ n_{\mathrm{max},\ell}
    \}_{\ell}` are the maximal number of radial wave numbers allowed in the
    specified range, and are indexed by :math:`(\ell, n)` tuples.  The
    normalisation coefficients derived from completeness relations are

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
    condition : {'Dirichlet', 'Neumann'}
        Either Dirichlet or Neumann boundary condition.
    cutoff : float
        Fourier scale upper cutoff.
    degmax : int or None, optional
        Maximum spherical degree (default is `None`).
    cuton : float, optional
        Fourier scale lower cutoff (default is 0.).
    degmin : int, optional
        Minimum spherical degree (default is 0).

    Attributes
    ----------
    degrees : int, array_like
        Spherical degrees associated with the discrete spectrum.
    depths : int, array_like
        Spectral depths associated with the discrete spectrum.
    roots : float, array_like
        Spherical Bessel roots associated with the discrete spectrum.
    mode_count : int
        Total number of allowed spectral modes, counting spherical order
        multuplicities.
    attrs : dict
        Discrete spectrum information, which contains the following keys:
        ``'min_wavenumber'``, ``'max_wavenumber'`` for minimum and maximum wave
        numbers :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}`;
        ``'boundary_radius'``, ``'bounded_volume'`` for the bounding radius and
        volume; ``'boundary_condition'`` for the boundary condition type.

    """

    _logger = logging.getLogger("DiscreteSpectrum")

    def __init__(self, radius, condition, cutoff, degmax=None, cuton=0.,
                 degmin=0):

        self.degrees, self.depths, self.roots, self.mode_count = \
            self.discretise(radius, condition, cuton, cutoff, degmin, degmax)

        self.attrs = {
            'min_wavenumber': cuton,
            'max_wavenumber': cutoff,
            'boundary_radius': radius,
            'bounded_volume': (4*np.pi/3) * radius**3,
            'boundary_condition': condition,
        }

        self._wavenumbers = None
        self._wave_tuples = None
        self._norm_coeff = None

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
        r"""Discrete wave numbers :math:`k_{\ell, n}`.

        Returns
        -------
        float, array_like
            Wave numbers.

        """
        if self._wavenumbers is not None:
            return self._wavenumbers

        self._wavenumbers = [
            u_ell / self.attrs['boundary_radius'] for u_ell in self.roots
        ]
        self._logger.info("Spectral wave numbers computed. ")

        return self._wavenumbers

    @property
    def dbl_indices(self):
        r"""Doublet index :math:`(\ell, n)` for each discrete wave number.

        Returns
        -------
        (int, int), array_like
            Doublet indices.

        """
        if self._wave_tuples is not None:
            return self._wave_tuples

        self._wave_tuples = [
            [
                (ell, nidx+1) for nidx in range(self.depths[ellidx])
            ]
            for ellidx, ell in enumerate(self.degrees)
        ]
        self._logger.info(
            "Doublet indices compiled for spectral wave numbers. "
        )

        return self._wave_tuples

    @property
    def normalisation(self):
        r"""Normalisation coefficient :math:`\kappa_{\ell n}` for each discrete
        wave number.

        Returns
        -------
        float, array_like
            Normalisation coefficients.

        """
        if self._norm_coeff is not None:
            return self._norm_coeff

        radius = self.attrs['boundary_radius']
        condition = self.attrs['boundary_condition'].lower()

        if condition.startswith('d'):
            self._norm_coeff = [
                2 / (radius**3 * spherical_besselj(ell+1, u_ell)**2)
                for ell, u_ell in zip(self.degrees, self.roots)
            ]
        elif condition.startswith('n'):
            self._norm_coeff = [
                2 / (radius**3 * spherical_besselj(ell, u_ell)**2) \
                    / (1 - ell*(ell+1)/np.square(u_ell))
                for ell, u_ell in zip(self.degrees, self.roots)
            ]
        self._logger.info(
            "Normalisations computed for spectral wave numbers. "
        )

        return self._norm_coeff

    @staticmethod
    def discretise(radius, condition, kmin, kmax, ellmin, ellmax):
        """
        Parameters
        ----------
        radius : float
            Boundary radius.
        condition : {'Dirichlet', 'Neumann'}
            Either Dirichlet or Neumann boundary condition.
        kmin, kmax : float
            Minimum and maximum wave numbers.
        ellmin, ellmax : int or None
            Minimum and maximum spherical degrees.

        Returns
        -------
        degrees : int, array_like
            Spherical function degrees.
        depths : int, array_like
            Maximum radial numbers, i.e. the number of allowed radial modes for
            each degree.
        roots : float, array_like
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

        degrees, roots, depths = [], [], []
        ell, mode_count = ellmin, 0
        while True:
            # Terminate if maximum `ell` passed.
            if ellmax is not None:
                if ell > ellmax:
                    _logger_.debug("Maximum degree reached. ")
                    break

            # Iterate through roots for the current `ell`.
            u_ell, n_ell = [], 0
            u = spherical_besselj_root(ell, n_ell+1, derivative=derivative)
            while kmin * radius <= u <= kmax * radius:
                u_ell.append(u)
                n_ell += 1
                u = spherical_besselj_root(ell, n_ell+1, derivative=derivative)

            # Terminate if no roots; else append results and add to counters.
            if n_ell == 0:
                _logger_.info("No more roots found. Last degree is %d. ", ell)
                break
            else:
                roots.append(np.asarray(u_ell))
                degrees.append(ell)
                depths.append(n_ell)
                mode_count += (2*ell + 1) * n_ell
                _logger_.debug("Results for degree %d appended. ", ell)

                ell += 1
                _logger_.debug("Moving on to next degree: %d. ", ell)

        return degrees, depths, roots, mode_count
