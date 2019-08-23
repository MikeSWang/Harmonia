"""
Spherical Fourier transform (:mod:`~harmonia.mapper.spherical_transform`)
===============================================================================

Transform discrete catalogues to discretised Fourier-space maps in spherical
coordinates.

.. todo:: Under development.

"""
import logging
import warnings

import numpy as np
from nbodykit.lab import FKPCatalog

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.morph import SphericalArray
from harmonia.algorithms.bases import sph_besselj, sph_harmonic
from harmonia.algorithms.integration import (
    angular_sphint_harmonic as aint_harmonic,
    radial_sphint_besselj as rint_besselj,
)
from harmonia.collections.utils import unitconst, cartesian_to_spherical as c2s
from .catalogue_maker import spherical_indicator as sphind


class SphericalMap:
    """Discretised spherical Fourier map.

    Attributes
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    data : :class:`nbodykit.base.catalog.CatalogSource`
        Data catalogue of particles.
    rand : :class:`nbodykit.base.catalog.CatalogSource` or None
        Random catalogue of particles.
    FKP : :class:`nbodykit.algorithms.convpower.catalog.FKPCatalog` or None
        FKP catalogue of both data and random particles.
    nmean : float
        Mean particle number density for the data catalogue.
    alpha_ratio : float or None
        Ratio of weighted mean particle number densities of the data catalogue
        to the random catalogue.

    """

    _logger = logging.getLogger('SphericalMap')

    def __init__(self, disc, data, rand=None, source='mock', nmean_data=None,
                 nmean_rand=None):
        """
        Parameters
        ----------
        disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
            Discrete spectrum.
        data : :class:`nbodykit.base.catalog.CatalogSource`
            Data catalogue of particles.
        rand : :class:`nbodykit.base.catalog.CatalogSource` or None, optional
            Random catalogue of particles (default is `None`).
        source : {'mock', 'survey'}, optional
            Catalogue source, either mock simulations (``'mock'``) or survey
            data (``'survey'``).
        nbar_data, nbar_rand : float or None, optional
            Input mean particle number density used to generate the data or the
            random catalogue (default is `None`).


        .. todo::

            * Implement for survey catalogue source.
            * Allow non-uniform redshift distribution in the FKP catalogue.

        """
        if np.min(disc.degrees) > 0:
            warnings.warn(
                "Angular modes up to degree {0} are missing. "
                "It is recommnded they be removed in post-processing instead. "
                .format(np.min(disc.degrees) - 1)
                )

        R, V = disc.attrs['boundary_radius'], disc.attrs['bounded_volume']

        if source == 'mock':
            centred_msg = (
                "Mock %s simulation box centred. "
                "Centred positions in 'Location' data column. "
                )
            inscribed_msg = (
                "Bounding sphere is not inscribed in %s catalogue: "
                f"boxsize %s; diameter {2*R}. "
                )
            sphericalised_msg = "Mock %s simulation box sphericalised. "
            boxsize_msg = (
                "Data and random catalogues have different "
                f"box sizes: %s and %s. "
                )
            intgrconstr_msg = "Integral constraint imposed. "

            # Compare Cartesian and spherical volumes.
            L_data = data.attrs['BoxSize']
            if not np.allclose(L_data/2, R):
                self._logger.info(inscribed_msg, "data", L_data)

            # Modify data catalogue.
            data['Location'] = data['Position'] - L_data/2
            self._logger.debug(centred_msg, "data")

            data['Selection'] *= sphind(data['Location'], R)
            self._logger.debug(sphericalised_msg, "data")

            # Modify random catalogue and/or compute `nmean` and `alpha_ratio`.
            if rand is None:
                if nmean_data is None:
                    nmean_data = np.sum(data['Selection'] * data['Weight']) / V
                    self._logger.warning(intgrconstr_msg)

                FKP = None
                nmean = float(nmean_data)
                alpha_ratio = None
            else:
                # Compare data and random catalogue box sizes.
                L_rand = rand.attrs['BoxSize']
                if not np.allclose(L_data, L_rand):
                    warnings.warn(boxsize_msg, L_data, L_rand)
                if not np.allclose(L_rand/2, R):
                    self._logger.info(inscribed_msg, "random", L_rand)

                rand['Location'] = rand['Position'] - L_rand/2
                self._logger.debug(centred_msg, "random")

                rand['Selection'] *= sphind(rand['Location'], R)
                self._logger.debug(sphericalised_msg, "random")

                if (nmean_data is None) or (nmean_rand is None):
                    nmean_data = np.sum(data['Selection'] * data['Weight']) / V
                    nmean_rand = np.sum(rand['Selection'] * rand['Weight']) / V
                    self._logger.warning(intgrconstr_msg)

                # FIXME: Correctly implement 'NZ' columns.
                data['NZ'] = nmean_data * data['Weight']
                rand['NZ'] = nmean_data * rand['Weight']

                FKP = FKPCatalog(data, rand)
                nmean = float(nmean_data)
                alpha_ratio = float(nmean_data/nmean_rand)

        self._n_ellmn = None
        self._nbar_ellmn = None

        self.disc = disc
        self.data = data
        self.rand = rand
        self.FKP = FKP
        self.nmean = nmean
        self.alpha_ratio = alpha_ratio

    @classmethod
    def discretise_catalogue(cls, data, rand=None, source='mock',
                             nmean_data=None, nmean_rand=None, **disc_params):
        """Instantiate the class from discretised spectrum.

        Parameters
        ----------
        disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
            Discrete spectrum.
        data : :class:`nbodykit.base.catalog.CatalogSource`
            Data catalogue of particles.
        rand : :class:`nbodykit.base.catalog.CatalogSource` or None, optional
            Random catalogue of particles (default is `None`).
        source : {'mock', 'survey'}, optional
            Catalogue source, either mock simulations (``'mock'``) or survey
            data (``'survey'``).
        nmean_data, nmean_rand : float or None, optional
            Input mean particle number density used to generate the data or the
            random catalogue (default is `None`).
        **disc_params
            Parameters as keyword arguments to be passed to
            :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`.

        """
        disc = DiscreteSpectrum(**disc_params)

        return cls(
            disc, data, rand=rand, source=source, nmean_data=nmean_data,
            nmean_rand=nmean_rand
            )

    def transform(self, method=None):
        """Perform discrete spherical Fourier transform.

        Parameters
        ----------
        method : {'sum', 'integrate'} or None, optional
            Computation method, either Monte Carlo (``'sum'``) or numerical
            integration (``'integrate'``) (default is `None`, but if not
            provided, this is subsequently overriden to ``'sum'`` if the random
            catalogue is not `None`, else set to ``'integrate'``).

        Returns
        -------
        n_ellmn, nbar_ellmn : nested list of complex, array_like
            Spherical Fourier coefficients for the observed and expected
            particle number densities, normalised to the homogeneous mean
            particle number density :attr:`nmean`.

        Raises
        ------
        ValueError
            If `method` is set to ``'sum'`` but :attr:`rand` is `None`.

        """
        # Get map information.
        data, rand = self.data, self.rand
        nbar, alpha = self.nmean, self.alpha_ratio
        R = self.disc.attrs['boundary_radius']
        ells, k_elln = self.disc.degrees, self.disc.wavenumbers

        # Fall to default method.
        method_msg = "Default method for computing expectation set to %s. "
        if method is None:
            if rand is None:
                method = 'integrate'
            else:
                method = 'sum'
            self._logger.info(method_msg, method)

        # Get data catalogue spherical positions, selections and weights.
        Dloc = c2s(data['Location'])
        Dsel = data['Selection']
        Dwgt = data['Weight']

        # Perform transforms.
        n_ellmn, nbar_ellmn = [], []
        for ell, k_ell in zip(ells, k_elln):
            n_l, nbar_l = [], []
            # Only compute for half the range thanks to parity.
            for m_l in range(-ell, 1):
                n_lm, nbar_lm = [], []
                for k_ln in k_ell:
                    # Evalute observed `n_lmn` by direct summation.
                    n_lmn = np.sum(
                        Dsel[:] * Dwgt[:]
                        * sph_besselj(ell, k_ln*Dloc[:, 0])
                        * np.conj(
                            sph_harmonic(ell, m_l, Dloc[:, 1], Dloc[:, 2])
                            )
                        )

                    # Evaluate expected `nbar_lmn` using `method`.
                    if method.lower() == 'sum':
                        if rand is None:
                            raise ValueError("Random catalogue missing. ")

                        # Get random catalogue spherical positions, selections
                        # and weights.
                        Rloc = c2s(rand['Location'])
                        Rsel = rand['Selection']
                        Rwgt = rand['Weight']

                        nbar_lmn = alpha * np.sum(
                            Rsel[:] * Rwgt[:]
                            * sph_besselj(ell, k_ln*Rloc[:, 0])
                            * np.conj(
                                sph_harmonic(ell, m_l, Rloc[:, 1], Rloc[:, 2])
                                )
                            )
                    elif method == 'integrate':
                        # FIXME: Only applies in the simplest scenario.
                        nbar_lmn = nbar \
                            * aint_harmonic(unitconst, ell, m_l) \
                            * rint_besselj(unitconst, ell, k_ln, R)

                    # `n_lmn`, `nbar_lmn` may be dask arrays.
                    n_lm.append(complex(n_lmn/nbar))
                    nbar_lm.append(complex(nbar_lmn/nbar))
                n_l.append(n_lm)
                nbar_l.append(nbar_lm)

            # Reflect and extend using parity.
            if ell != 0:
                n_l_ud = np.multiply(
                    (-1)**np.arange(1, ell+1)[:, None], np.flipud(n_l[:-1])
                    )
                nbar_l_ud = np.multiply(
                    (-1)**np.arange(1, ell+1)[:, None], np.flipud(nbar_l[:-1])
                    )
                n_l = np.concatenate((n_l, np.conj(n_l_ud)))
                nbar_l = np.concatenate((nbar_l, np.conj(nbar_l_ud)))

            n_ellmn.append(np.asarray(n_l))
            nbar_ellmn.append(np.asarray(nbar_l))

        self._n_ellmn, self._nbar_ellmn = n_ellmn, nbar_ellmn

        return n_ellmn, nbar_ellmn

    def twopoint(self, method=None, pivot='natural', ord_collapse=False):
        r"""Comptute 2-point statistics.

        Parameters
        ----------
        method : str, optional
            Expectation computation method (default is `None`).
        pivot : {'natural', 'scale', 'lmn', 'lnm', 'nlm', 'ln', 'k'}, optional
            Axis order for array flattening (default is ``'natural'``).  See
            also :class:`~harmonia.algorithms.morphing.SphericalArray`.
        ord_collapse : bool, optional
            If `True`, spherical Fourier coefficients are first averaged over
            spherical orders.  See also
            :class:`~harmonia.algorithms.morphing.SphericalArray`.

        Returns
        -------
        list of complex, array_like
            2-point statistics as 2-d array.

        Warnings
        --------
        For this method, the spherical degrees are assumed to start at
        :math:`\ell = 0`.  See :ref:`this note <ellidx-warning>` for
        :mod:`~harmonia.reader.spherical_model`.

        """
        if (self._n_ellmn is None) or (self._nbar_ellmn is None):
            self._n_ellmn, self._nbar_ellmn = self.transform(method=method)

        return self.compute_stat2pt_from_coeff(
            self._n_ellmn, self._nbar_ellmn, self.disc, pivot=pivot,
            ord_collapse=ord_collapse
            )

    @staticmethod
    def compute_stat2pt_from_coeff(n_coeff, nbar_coeff, disc, pivot='natural',
                                   ord_collapse=False):
        r"""Compute 2-point statistics from spherical Fourier coefficients.

        Parameters
        ----------
        n_coeff, nbar_coeff : nested list of complex, array_like
            Observed and expected pherical Fourier coefficients of the field
            normalised to the homogeneous particle number density
            :attr:`nmean`.
        disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
            Discrete spectrum.
        pivot : {'natural', 'scale', 'lmn', 'lnm', 'nlm', 'ln', 'k'}, optional
            Axis order for array flattening (default is ``'natural'``).  See
            also :class:`~harmonia.algorithms.morphing.SphericalArray`.
        ord_collapse : bool, optional
            If `True`, coefficients `n_coeff` , `nbar_coeff` are first averaged
            over spherical orders.  See also
            :class:`~harmonia.algorithms.morphing.SphericalArray`.

        Returns
        -------
        list of complex, array_like
            2-point statistics as 2-d array.

        Warnings
        --------
        For this method, the spherical degrees are assumed to start at
        :math:`\ell = 0`.  See :ref:`this note <ellidx-warning>` for
        :mod:`~harmonia.reader.spherical_model`.

        """
        fill = [n - nbar for n, nbar in zip(n_coeff, nbar_coeff)]

        d_ellmn = SphericalArray.build(disc=disc, filling=fill)
        dvec, _ = d_ellmn.unfold(pivot, collapse=ord_collapse)

        return np.outer(dvec, np.conj(dvec))

    def spherical_power(self, method=None):
        """Compute spherically recovered power spectrum.

        Parameters
        ----------
        method : str, optional
            Expectation computation method (default is `None`).

        Returns
        -------
        sph_power : float, array_like
            Spherically recovered power.

        """
        if (self._n_ellmn is None) or (self._nbar_ellmn is None):
            self._n_ellmn, self._nbar_ellmn = self.transform(method=method)

        sph_power = self.square_amplitude(
            self._n_ellmn, self._nbar_ellmn, normcoeff=self.disc.normcoeff
            )

        return sph_power

    @staticmethod
    def square_amplitude(n_coeff, nbar_coeff, normcoeff=None):
        """Compute normalised square_amplitude from spherical Fourier
        coefficients of the field.

        Parameters
        ----------
        n_coeff, nbar_coeff : nested list of complex, array_like
            Observed and expected spherical Fourier coefficients for the field
            normalised to the homogeneous particle number density
            :attr:`nmean`.
        normcoeff : list of float, array_like, optional
            Normalisation coefficients (default is `None`).

        Returns
        -------
        float, array_like
            Spherically recovered power with given normalisation.

        Notes
        -----
        If `normcoeff` is `None`, all normalisation coefficients are set to
        unity.

        """
        if normcoeff is None:
            normcoeff = np.ones(len(n_coeff))

        return [
            norm * np.average(np.abs(n - nbar)**2, axis=0)
            for norm, n, nbar in zip(normcoeff, n_coeff, nbar_coeff)
            ]
