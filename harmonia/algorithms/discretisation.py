"""
Spectrum discretisation (:mod:`~harmonia.algorithms.discretisation`)
===============================================================================

Discretise Fourier spectrum for analysing cosmological fields.

.. autosummary::

    DiscreteSpectrum

|

"""
import logging

import numpy as np

from .bases import sph_besselj, sph_besselj_root


class DiscreteSpectrum:
    r"""Discrete Fourier spectrum for given radial boundary conditions.

    When a boundary condition is prescribed at some maximum radius
    :math:`r = R`, the allowed wave numbers for the discretised
    spectrum are :math:`k_{\ell n} = u_{\ell n} / R` where
    :math:`\{ u_{\ell n}: n = 1, \dots, n_{\mathrm{max},\ell} \}_{\ell}` are
    roots of the spherical Bessel functions of order :math:`\ell` if the
    boundary condition is Dirichlet, or their derivative functions if the
    boundary condition is Neumann.  :math:`\{ n_{\mathrm{max},\ell} \}_{\ell}`,
    known as the spherical depths, are the maximal number of radial modes
    allowed in the specified range of spectral wave numbers, which are indexed
    by :math:`\{ (\ell, n) \}` tuples.  The normalisation coefficients derived
    from completeness relations are

    .. math::

        \kappa_{\ell n} = \frac{2}{R^3} j_{\ell+1}^{-2}(u_{\ell n}) \,,

    for Dirichlet boundary conditions, and

    .. math::

        \kappa_{\ell n} = \frac{2}{R^3} j_{\ell}^{-2}(u_{\ell n})
        \Big[ 1 - \ell(\ell+1) / u_{\ell n}^2 \Big]^{-1} \,,

    for Neumann boundary conditions.

    Note the length unit is Mpc/h.

    Attributes
    ----------
    degrees : int, array_like
        Spherical degrees associated with the discrete spectrum.
    depths : int, array_like
        Spectral depths associated with the discrete spectrum.
    roots : float, array_like
        Spherical Bessel roots associated with the discrete spectrum.
    nmodes : int
        Total number of allowed spectral modes.
    attrs : dict
        Discrete spectrum information.  Contains the following keys:
        ``'maxscale'``, ``'minscale'`` for minimum and maximum wave numbers
        :math:`k_\mathrm{min}` and :math:`k_\mathrm{max}`;
        ``'boundary_radius'``, ``'bounded_volume'`` for the bounding radius and
        volume; ``'boundary_condition'`` for the boundary condition type.

    """

    _logger = logging.getLogger("DiscreteSpectrum")

    def __init__(self, radius, condition, cutoff, degmax=None, cuton=0.,
                 degmin=0):
        """
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

        """
        self.degrees, self.depths, self.roots, self.nmodes = self.discretise(
            radius, condition, cuton, cutoff, degmin, degmax
            )

        self.attrs = {
            'maxscale': cuton,
            'minscale': cutoff,
            'boundary_radius': radius,
            'bounded_volume': (4*np.pi/3) * radius**3,
            'boundary_condition': condition,
            }

        self._wavenumbers = None
        self._waveindices = None
        self._normcoeff = None

        self._logger.info(
            "%s computed: %d degrees and %d modes in total. ",
            self.__repr__(), len(self.degrees), self.nmodes
            )

    def __repr__(self):
        return "Spectrum({0}, radius={1}, {2} <= wavenum <= {3})".format(
            self.attrs['boundary_condition'], self.attrs['boundary_radius'],
            self.attrs['maxscale'], self.attrs['minscale']
            )

    @property
    def wavenumbers(self):
        """Discrete wave numbers.

        Returns
        -------
        float, array_like
            Discrete wave numbers.

        """
        if self._wavenumbers is not None:
            return self._wavenumbers

        self._wavenumbers = [
            u_ell / self.attrs['boundary_radius'] for u_ell in self.roots
            ]

        self._logger.info("Spectral wave numbers computed. ")

        return self._wavenumbers

    @property
    def waveindices(self):
        """Doublet indices for each discrete wave number.

        Returns
        -------
        (int, int), array_like
            Spectral mode indices.

        """
        if self._waveindices is not None:
            return self._waveindices

        self._waveindices = [
            [(ell, nidx+1) for nidx in range(self.depths[ellidx])]
            for ellidx, ell in enumerate(self.degrees)
            ]
        self._logger.info("Spectral mode indices compiled. ")

        return self._waveindices

    @property
    def normcoeff(self):
        """Normalisation coefficients for discretisation.

        Returns
        -------
        kappa_elln : float, array_like
            Normalisation coefficients.

        """
        if self._normcoeff is not None:
            return self._normcoeff

        R = self.attrs['boundary_radius']
        cond = self.attrs['boundary_condition'].lower()

        kappa_elln = []
        if cond.startswith('d'):
            for ell, u_ell in zip(self.degrees, self.roots):
                kappa_elln.append(
                    2 / (R**3 * sph_besselj(ell+1, u_ell)**2)
                    )
        elif cond.startswith('n'):
            for ell, u_ell in zip(self.degrees, self.roots):
                kappa_elln.append(
                    2 / (R**3 * sph_besselj(ell, u_ell)**2)
                    / (1 - ell*(ell+1)/np.square(u_ell))
                    )

        self._normcoeff = kappa_elln
        self._logger.info("Spectral normalisations computed. ")

        return kappa_elln

    @staticmethod
    def discretise(R, condition, kmin, kmax, ellmin, ellmax):
        """
        Parameters
        ----------
        R : float
            Boundary radius.
        condition : {'Dirichlet', 'Neumann'}
            Either Dirichlet or Neumann boundary condition.
        kmin, kmax : float
            Minimum and maximum wave numbers.
        ellmin, ellmax : int or None
            Minimum and maximum spherical degrees.

        Returns
        -------
        ells : int, array_like
            Spherical function degrees.
        nmaxs : int, array_like
            Maximum radial numbers, i.e. the number of allowed spectral modes
            for each degree.
        us : float, array_like
            Spherical Bessel roots.
        ntotal : int
            Total number of spectral modes.

        Raises
        ------
        ValueError
            If `condition` does not correpond to either 'Dirichlet' or
            'Neumann'.

        """
        _logger_ = logging.getLogger('discretisation')

        # Whether derivative is taken of the spherical Bessel function.
        if condition.lower().startswith('d'):
            deriv = False
        elif condition.lower().startswith('n'):
            deriv = True
        else:
            raise ValueError(f"Invalid boundary condition type: {condition}. ")

        # Initialise arrays and counters.
        ells, us, nmaxs = [], [], []
        ell, ntotal = ellmin, 0
        while True:
            # Terminate if maximum `ell` passed.
            if ellmax is not None:
                if ell > ellmax:
                    _logger_.debug("Maximum degree reached. ")
                    break

            # Iterate through roots for the current `ell`.
            u_ell, n_ell = [], 0
            u = sph_besselj_root(ell, n_ell+1, deriv=deriv)
            while kmin*R <= u <= kmax*R:
                u_ell.append(u)
                n_ell += 1
                u = sph_besselj_root(ell, n_ell+1, deriv=deriv)

            # Terminate if no roots; else append results and add to counters.
            if n_ell == 0:
                _logger_.info("No more roots found. Last degree is %d. ", ell)
                break
            else:
                us.append(np.asarray(u_ell))
                ells.append(ell)
                nmaxs.append(n_ell)
                _logger_.debug("Results for degree %d appended. ", ell)

                ntotal += (2*ell+1) * n_ell
                ell += 1
                _logger_.debug("Moving on to next degree: %d. ", ell)

        return ells, nmaxs, us, ntotal
