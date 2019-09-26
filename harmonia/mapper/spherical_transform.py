"""
Spherical Fourier transform (:mod:`~harmonia.mapper.spherical_transform`)
===============================================================================

Transform discrete catalogues to discretised Fourier-space maps in spherical
coordinates.

"""
import logging
import warnings

import numpy as np
from nbodykit.lab import FKPCatalog

from .catalogue_maker import spherical_indicator as spherical_cut
from harmonia.algorithms._bases import spherical_besselj, spherical_harmonic
from harmonia.algorithms._integration import (
    angular_harmonic_integral as aint_harmonic,
    radial_besselj_integral as rint_besselj,
)
from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.morph import SphericalArray
from harmonia.collections.utils import (
    unit_const,
    cartesian_to_spherical as c2s,
)


class SphericalMap:
    """Discretised spherical Fourier map from catalogue sources.

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    data : :class:`nbodykit.base.catalog.CatalogSource`
        Data catalogue of particles.
    rand : :class:`nbodykit.base.catalog.CatalogSource` or None, optional
        Random catalogue of particles (default is `None`).
    source : {'mock', 'survey'}, optional
        Catalogue source, either ``'mock'`` simulations or ``'survey'`` data.
    mean_density_data, mean_density_rand : float or None, optional
        Input mean particle number density (in cubic h/Mpc) of the data and/or
        random catalogue (default is `None`).

    Attributes
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    data : :class:`nbodykit.base.catalog.CatalogSource`
        Data catalogue of particles.
    rand : :class:`nbodykit.base.catalog.CatalogSource` or None
        Random catalogue of particles.
    pair : :class:`nbodykit.algorithms.convpower.catalog.FKPCatalog` or None
        FKP pair of data and random catalogues.
    mean_density : float
        Mean particle number density (in cubic h/Mpc) for the data catalogue.
    alpha_ratio : float or None
        Ratio of weighted mean particle number densities of the data catalogue
        to that of the random catalogue.

    """

    _logger = logging.getLogger('SphericalMap')
    _msg = {
        'boxsizes': (
            "Data and random catalogues have different box sizes: "
            "%.1f and %.1f. "
        ),
        'centering': (
            "Mock %s simulation box centred. "
            "Centred positions stored in 'Location' column. "
        ),
        'inscribing': (
            "Bounding sphere is not inscribed in %s catalogue: "
            "boxsize %.1f; diameter %.1f. "
        ),
        'integral_constraint': "Integral constraint imposed. ",
        'method': "Default method for computing expectation set to %s. ",
        'spherical_cut': "Mock %s simulation box cut to sphere. ",
    }

    def __init__(self, disc, data, rand=None, source='mock',
                 mean_density_data=None, mean_density_rand=None):

        radius = disc.attrs['boundary_radius']
        volume = disc.attrs['bounded_volume']

        self.disc = disc

        if np.min(disc.degrees) != 0:
            warnings.warn(
                "Angular modes up to degree {0} are missing. "
                "It is recommnded they be removed in post-processing instead. "
                .format(np.min(disc.degrees) - 1)
            )

        if source == 'mock':
            data_boxsize = data.attrs['BoxSize']
            if not np.allclose(data_boxsize/2, radius):
                self._logger.info(
                    self._msg['inscribing'],
                    "data",
                    data_boxsize,
                    2*radius,
                )

            data['Location'] = data['Position'] - data_boxsize/2
            self._logger.debug(self._msg['centering'], "data")

            data['Selection'] *= spherical_cut(data['Location'], radius)
            self._logger.debug(self._msg['spherical_cut'], "data")

            if rand is None:
                if mean_density_data is None:
                    mean_density_data = np.sum(
                        data['Selection'] * data['Weight']
                    ) / volume
                    self._logger.warning(self._msg['integral_constraint'])
                mean_density = float(mean_density_data)

                pair = None
                alpha_ratio = None
            else:
                rand_boxsize = rand.attrs['BoxSize']
                if not np.allclose(data_boxsize, rand_boxsize):
                    warnings.warn(
                        self._msg['boxsizes'],
                        data_boxsize,
                        rand_boxsize,
                    )
                if not np.allclose(rand_boxsize/2, radius):
                    self._logger.info(
                        self._msg['inscribing'],
                        "random",
                        rand_boxsize,
                        2*radius,
                    )

                rand['Location'] = rand['Position'] - rand_boxsize/2
                self._logger.debug(self._msg['centering'], "random")

                rand['Selection'] *= spherical_cut(rand['Location'], radius)
                self._logger.debug(self._msg['spherical_cut'], "random")

                if mean_density_data is None or mean_density_rand is None:
                    mean_density_data = np.sum(
                        data['Selection'] * data['Weight'],
                    ) / volume
                    mean_density_rand = np.sum(
                        rand['Selection'] * rand['Weight'],
                    ) / volume
                    self._logger.warning(self._msg['integral_constraint'])
                mean_density = float(mean_density_data)

                # FIXME: Correctly implement 'NZ' columns.
                data['NZ'] = mean_density_data * data['Weight']
                rand['NZ'] = mean_density_data * rand['Weight']

                pair = FKPCatalog(data, rand)
                alpha_ratio = float(mean_density_data/mean_density_rand)
        # FIXME: Implement for survey input data.
        elif source == 'survey':
            raise NotImplementedError

        self.data = data
        self.rand = rand
        self.mean_density = mean_density
        self.pair = pair
        self.alpha_ratio = alpha_ratio

        self._n_coeff = None
        self._nbar_coeff = None

    def __str__(self):
        return "SphericalMap(maxdegree={}, modecount={})".format(
            max(self.disc.degrees),
            self.disc.mode_count,
        )

    @classmethod
    def discretise_catalogue(cls, data, rand=None, source='mock',
                             mean_density_data=None, mean_density_rand=None,
                             **disc_params):
        """Instantiate a spherical map from discretised spectrum parameters.

        Parameters
        ----------
        disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
            Discrete spectrum.
        data : :class:`nbodykit.base.catalog.CatalogSource`
            Data catalogue of particles.
        rand : :class:`nbodykit.base.catalog.CatalogSource` or None, optional
            Random catalogue of particles (default is `None`).
        source : {'mock', 'survey'}, optional
            Catalogue source, either ``'mock'`` simulations or ``'survey'``
            data.
        mean_density_data, mean_density_rand : float or None, optional
            Input mean particle number density (in cubic h/Mpc) of the data or
            random catalogue (default is `None`).
        **disc_params
            Parameters as keyword arguments to be passed to
            :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`.

        """
        return cls(
            DiscreteSpectrum(**disc_params),
            data,
            rand=rand,
            source=source,
            mean_density_data=mean_density_data,
            mean_density_rand=mean_density_rand,
            )

    def transform(self, method=None):
        """Perform discrete spherical Fourier transform.

        Parameters
        ----------
        method : {'sum', 'integrate'} or None, optional
            Computation method, either Monte Carlo (``'sum'``) or numerical
            integration (``'integrate'``) (default is `None`; if not provided,
            this is subsequently overriden to ``'sum'`` if the random catalogue
            is not `None`, else set to ``'integrate'``).

        Returns
        -------
        n_coeff, nbar_coeff : nested list of complex, array_like
            Spherical Fourier coefficients for the observed and expected
            particle number densities, normalised to the homogeneous mean
            particle number density :attr:`mean_density`.

        Raises
        ------
        ValueError
            If `method` is set to ``'sum'`` but :attr:`rand` is `None`.

        """
        data, rand = self.data, self.rand
        nbar, alpha = self.mean_density, self.alpha_ratio
        radius = self.disc.attrs['boundary_radius']

        if method is None:
            if rand is None:
                method = 'integrate'
            else:
                method = 'sum'
            self._logger.info(self._msg['method'], method)

        loc_data = c2s(data['Location'])
        sel_data = data['Selection']
        wgt_data = data['Weight']

        n_coeff, nbar_coeff = [], []
        for ell, k_ell in zip(self.disc.degrees, self.disc.wavenumbers):
            n_ell, nbar_ell = [], []
            for m_ell in range(-ell, 1):  # half of computation using parity
                n_ellm, nbar_ellm = [], []
                for k_elln in k_ell:
                    n_ellmn = np.sum(
                        sel_data[:] * wgt_data[:]
                        * spherical_besselj(ell, k_elln*loc_data[:, 0])
                        * np.conj(
                            spherical_harmonic(
                                ell,
                                m_ell,
                                loc_data[:, 1],
                                loc_data[:, 2],
                            )
                        )
                    )

                    if method.lower() == 'sum':
                        if rand is None:
                            raise ValueError("Random catalogue missing. ")

                        loc_rand = c2s(rand['Location'])
                        sel_rand = rand['Selection']
                        wgt_rand = rand['Weight']

                        nbar_ellmn = alpha * np.sum(
                            sel_rand[:] * wgt_rand[:]
                            * spherical_besselj(ell, k_elln*loc_rand[:, 0])
                            * np.conj(
                                spherical_harmonic(
                                    ell,
                                    m_ell,
                                    loc_rand[:, 1],
                                    loc_rand[:, 2],
                                )
                            )
                        )
                    elif method == 'integrate':
                        # FIXME: Only applies without selection function.
                        nbar_ellmn = nbar \
                            * aint_harmonic(unit_const, ell, m_ell) \
                            * rint_besselj(unit_const, ell, k_elln, radius)

                    # `n_ellmn`, `nbar_ellmn` may be dask arrays.
                    n_ellm.append(complex(n_ellmn/nbar))
                    nbar_ellm.append(complex(nbar_ellmn/nbar))
                n_ell.append(n_ellm)
                nbar_ell.append(nbar_ellm)

            if ell != 0:  # reflect and extend using parity
                n_ell_flip = np.multiply(
                    (-1)**np.arange(1, ell+1)[:, None],
                    np.flipud(n_ell[:-1]),
                )
                nbar_ell_flip = np.multiply(
                    (-1)**np.arange(1, ell+1)[:, None],
                    np.flipud(nbar_ell[:-1]),
                )
                n_ell = np.concatenate((n_ell, np.conj(n_ell_flip)))
                nbar_ell = np.concatenate((nbar_ell, np.conj(nbar_ell_flip)))

            n_coeff.append(np.asarray(n_ell))
            nbar_coeff.append(np.asarray(nbar_ell))

        self._n_coeff, self._nbar_coeff = n_coeff, nbar_coeff

        return n_coeff, nbar_coeff

    def two_point(self, method=None, pivot='natural', order_collapse=False):
        r"""Comptute 2-point statistics.

        See :class:`~harmonia.algorithms.morph.SphericalArray` for array
        structure.

        Parameters
        ----------
        method : str, optional
            Expectation computation method (default is `None`).
        pivot : {'natural', 'scale', 'lmn', 'lnm', 'nlm', 'ln', 'k'}, optional
            Axis order for array flattening (default is ``'natural'``).
        order_collapse : bool, optional
            If `True` (default is `False`), spherical Fourier coefficients are
            first averaged over spherical orders.

        Returns
        -------
        list of complex, array_like
            2-point statistics as 2-d array.

        Notes
        -----
        For this method, the spherical degrees are assumed to start at
        :math:`\ell = 0`.  See :ref:`this note <degree-index-warning>` for
        :mod:`~harmonia.reader.spherical_model`.

        """
        if self._n_coeff is None or self._nbar_coeff is None:
            self._n_coeff, self._nbar_coeff = self.transform(method=method)

        return self.compute_two_point_from_coeff(
            self._n_coeff,
            self._nbar_coeff,
            self.disc,
            pivot=pivot,
            order_collapse=order_collapse,
        )

    def spherical_power(self, method=None):
        """Compute spherically recovered power spectrum.

        Parameters
        ----------
        method : str, optional
            Expectation computation method (default is `None`).

        Returns
        -------
        spherical_power : float, array_like
            Spherically recovered power.

        """
        if self._n_coeff is None or self._nbar_coeff is None:
            self._n_coeff, self._nbar_coeff = self.transform(method=method)

        spherical_power = self.square_amplitude(
            self._n_coeff,
            self._nbar_coeff,
            normalisation=self.disc.normalisation,
        )

        return spherical_power

    @staticmethod
    def compute_two_point_from_coeff(n_coeff, nbar_coeff, disc,
                                     pivot='natural', order_collapse=False):
        r"""Compute 2-point statistics from spherical Fourier coefficients.

        See :class:`~harmonia.algorithms.morph.SphericalArray` for array
        structure.

        Parameters
        ----------
        n_coeff, nbar_coeff : nested list of complex, array_like
            Observed and expected pherical Fourier coefficients of the field
            normalised to the homogeneous particle number density
            :attr:`mean_density`.
        disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
            Discrete spectrum.
        pivot : {'natural', 'scale', 'lmn', 'lnm', 'nlm', 'ln', 'k'}, optional
            Axis order for array flattening (default is ``'natural'``).
        order_collapse : bool, optional
            If `True`, coefficients `n_coeff` , `nbar_coeff` are first averaged
            over spherical orders.

        Returns
        -------
        list of complex, array_like
            2-point statistics as 2-d array.

        Warnings
        --------
        For this method, the spherical degrees are assumed to start at
        :math:`\ell = 0`.  See :ref:`this note <degree-index-warning>` for
        :mod:`~harmonia.reader.spherical_model`.

        """
        fill = [n - nbar for n, nbar in zip(n_coeff, nbar_coeff)]

        delta_ellmn = SphericalArray.build(disc=disc, filling=fill)
        delta_ellmn_flat = delta_ellmn.unfold(
            pivot,
            collapse=order_collapse,
            return_only='data',
        )

        return np.outer(delta_ellmn_flat, np.conj(delta_ellmn_flat))

    @staticmethod
    def square_amplitude(n_coeff, nbar_coeff, normalisation=None):
        """Compute normalised square amplitude from spherical Fourier
        coefficients of the field.

        Parameters
        ----------
        n_coeff, nbar_coeff : nested list of complex, array_like
            Observed and expected spherical Fourier coefficients for the field
            normalised to the homogeneous particle number density
            :attr:`mean_density`.
        normalisation : list of float, array_like, optional
            Normalisation coefficients (default is `None`).

        Returns
        -------
        float, array_like
            Spherically recovered power with given normalisation.

        Notes
        -----
        If `normalisation` is `None`, all normalisation coefficients are set to
        unity.

        """
        if normalisation is None:
            normalisation = np.ones(len(n_coeff))

        return [
            kappa * np.average(np.abs(n_ellmn - nbar_ellmn)**2, axis=0)
            for kappa, n_ellmn, nbar_ellmn in zip(
                normalisation,
                n_coeff,
                nbar_coeff,
            )
        ]
