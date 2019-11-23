"""
Spherical Fourier transform (:mod:`~harmonia.mapper.spherical_transform`)
===========================================================================

Transform discrete catalogues to discretised Fourier-space maps in
spherical coordinates.

.. autosummary::

    SphericalMap

|

"""
import logging
import warnings

import numpy as np
from nbodykit.lab import FKPCatalog

from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic
from harmonia.algorithms.integration import \
    angular_harmonic_integral as ang_int_harmonic
from harmonia.algorithms.integration import \
    radial_besselj_integral as rad_int_besselj
from harmonia.algorithms.morph import SphericalArray
from harmonia.collections.utils import cartesian_to_spherical as c2s
from harmonia.collections.utils import spherical_indicator as spherical_cut
from harmonia.collections.utils import unit_const


class SphericalMap:
    r"""Discretised spherical Fourier map from catalogue sources.

    Notes
    -----
    The spherical degrees of the map is usually assumed to start at
    :math:`\ell = 0` (see :ref:`warning <degree-index-warning>` in
    :mod:`~harmonia.reader.spherical_model`).

    Parameters
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    data : :class:`nbodykit.base.catalog.CatalogSource`
        Data catalogue of particles.
    rand : :class:`nbodykit.base.catalog.CatalogSource` *or None, optional*
        Random catalogue of particles (default is `None`).
    mean_density_data, mean_density_rand : float or None, optional
        Mean particle number density (in cubic h/Mpc) of the data or
        random catalogue (default is `None`).
    source : {'simulation', 'survey'}, optional
        Catalogue source, either ``'simulation'`` for simulations or
        ``'survey'`` data.

    Attributes
    ----------
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum.
    data : :class:`nbodykit.base.catalog.CatalogSource`
        Data catalogue of particles.
    rand : :class:`nbodykit.base.catalog.CatalogSource` *or None*
        Random catalogue of particles.
    pair : |fkp_catalogue| *or None*
        FKP pair of data and random catalogues.
    mean_density : float
        Mean particle number density (in cubic h/Mpc) of the data
        catalogue.
    alpha_ratio : float or None
        Ratio of weighted mean particle number densities of the data
        catalogue to that of the random catalogue.


    .. |fkp_catalogue| replace::

        :class:`nbodykit.algorithms.convpower.catalog.FKPCatalog`

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

    def __init__(self, disc, data, rand=None, mean_density_data=None,
                 mean_density_rand=None, source='simulation'):

        self.disc = disc
        if np.min(disc.degrees) > 0:
            warnings.warn(
                "Fourier modes up to degree {0} are missing. "
                "It is recommnded they be removed in post-processing instead. "
                .format(np.min(disc.degrees) - 1)
            )

        radius = disc.attrs['boundary_radius']
        volume = disc.attrs['bounded_volume']

        if source == 'simulation':
            data_boxsize = data.attrs['BoxSize']
            if not np.allclose(data_boxsize, 2*radius):
                self._logger.info(
                    self._msg['inscribing'], "data", data_boxsize, 2*radius
                )

            data['Location'] = data['Position'] \
                - np.divide(data_boxsize, 2)
            self._logger.debug(self._msg['centering'], "data")

            data['Selection'] *= spherical_cut(data['Location'], radius)
            self._logger.debug(self._msg['spherical_cut'], "data")

            if rand is None:
                if mean_density_data is None:
                    mean_density_data = 1 / volume \
                        * np.sum(data['Selection'] * data['Weight'])
                    self._logger.warning(self._msg['integral_constraint'])
                mean_density = float(mean_density_data)

                pair = None
                alpha_ratio = None
            else:
                rand_boxsize = rand.attrs['BoxSize']
                if not np.allclose(data_boxsize, rand_boxsize):
                    warnings.warn(
                        self._msg['boxsizes'], data_boxsize, rand_boxsize
                    )
                if not np.allclose(rand_boxsize, 2*radius):
                    self._logger.info(
                        self._msg['inscribing'],
                        "random", rand_boxsize, 2*radius
                    )

                rand['Location'] = rand['Position'] \
                    - np.divide(rand_boxsize, 2)
                self._logger.debug(self._msg['centering'], "random")

                rand['Selection'] *= spherical_cut(rand['Location'], radius)
                self._logger.debug(self._msg['spherical_cut'], "random")

                if mean_density_data is None or mean_density_rand is None:
                    mean_density_data = 1 / volume \
                        * np.sum(data['Selection'] * data['Weight'])
                    mean_density_rand = 1 / volume \
                        * np.sum(rand['Selection'] * rand['Weight'])
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

        return "SphericalMap({}, id={})".format(repr(self.disc), id(self))

    def transform(self, method=None):
        """Perform discrete spherical Fourier transform by direct
        summation.

        Parity relations between spherical harmonics of opposite orders but
        the same degree are employed to reduce computational effort.

        Parameters
        ----------
        method : {'sum', 'integrate', None}, optional
            Computation method for expection, either Monte Carlo over
            the random catalogue (``'sum'``) or numerical integration
            (``'integrate'``).  If `None` (default), this may be
            subsequently overriden to ``'sum'`` if the random catalogue
            is not `None`, else set to ``'integrate'``.

        Returns
        -------
        n_coeff, nbar_coeff : dict of {int: complex :class:`numpy.ndarray`}
            Spherical Fourier coefficients for the observed and expected
            particle number densities, normalised to the homogeneous mean
            particle number density :attr:`mean_density`.

        Raises
        ------
        ValueError
            If `method` is set to ``'sum'`` but :attr:`rand` is `None`.

        """
        if method is None:
            if self.rand is None:
                method = 'integrate'
            else:
                method = 'sum'
            self._logger.info(self._msg['method'], method)

        loc_data = c2s(self.data['Location'])
        sel_data = self.data['Selection']
        wgt_data = self.data['Weight']

        n_coeff, nbar_coeff = {}, {}
        for ell in self.disc.degrees:
            n_ell, nbar_ell = [], []
            for m_ell in range(-ell, 1):
                n_ellm, nbar_ellm = [], []
                for k_elln in self.disc.wavenumbers[ell]:
                    harm_args = ell, m_ell, loc_data[:, 1], loc_data[:, 2]
                    n_ellmn = np.sum(
                        sel_data[:] * wgt_data[:]
                        * spherical_besselj(ell, k_elln*loc_data[:, 0])
                        * np.conj(spherical_harmonic(*harm_args))
                    )

                    if method.lower() == 'sum':
                        if self.rand is None:
                            raise ValueError("Random catalogue missing. ")

                        loc_rand = c2s(self.rand['Location'])
                        sel_rand = self.rand['Selection']
                        wgt_rand = self.rand['Weight']

                        harm_args = ell, m_ell, loc_rand[:, 1], loc_rand[:, 2]
                        nbar_ellmn = self.alpha_ratio * np.sum(
                            sel_rand[:] * wgt_rand[:]
                            * spherical_besselj(ell, k_elln*loc_rand[:, 0])
                            * np.conj(spherical_harmonic(*harm_args))
                        )
                    elif method == 'integrate':
                        # FIXME: Only applies without selection function.
                        nbar_ellmn = self.mean_density \
                            * ang_int_harmonic(unit_const, ell, m_ell) \
                            * rad_int_besselj(
                                unit_const, ell, k_elln,
                                self.disc.attrs['boundary_radius']
                            )

                    # CAVEAT: `n_ellmn`, `nbar_ellmn` may be dask arrays.
                    n_ellm.append(complex(n_ellmn/self.mean_density))
                    nbar_ellm.append(complex(nbar_ellmn/self.mean_density))
                n_ell.append(n_ellm)
                nbar_ell.append(nbar_ellm)

            if ell != 0:
                n_ell_parity = np.conj(
                    np.power(-1, np.arange(1, ell+1)[:, None])
                    * np.flipud(n_ell[:-1])
                )
                nbar_ell_parity = np.conj(
                    np.power(-1, np.arange(1, ell+1)[:, None])
                    * np.flipud(nbar_ell[:-1])
                )

                n_ell = np.concatenate((n_ell, n_ell_parity))
                nbar_ell = np.concatenate((nbar_ell, nbar_ell_parity))

            n_coeff[ell] = np.array(n_ell)
            nbar_coeff[ell] = np.array(nbar_ell)

        self._n_coeff, self._nbar_coeff = n_coeff, nbar_coeff

        return n_coeff, nbar_coeff

    def density_constrast(self, method=None):
        """Compute spherical Fourier density contrast coefficients.

        Parameters
        ----------
        method : str, optional
            Expectation computation method (default is `None`) if the
            transformed spherical Fourier coefficients have not been
            computed.

        Returns
        -------
        density_constrast : dict of {int: complex :class:`numpy.ndarray`}
            Spherical Fourier density contrast coefficients.

        """
        if self._n_coeff is None or self._nbar_coeff is None:
            self._n_coeff, self._nbar_coeff = self.transform(method=method)

        density_constrast = {
            ell: self._n_coeff[ell] - self._nbar_coeff[ell]
            for ell in self.disc.degrees
        }

        return density_constrast

    def spherical_power(self, method=None):
        """Compute spherically recovered power spectrum.

        Parameters
        ----------
        method : str, optional
            Expectation computation method (default is `None`) if the
            transformed spherical Fourier coefficients have not been
            computed.

        Returns
        -------
        spherical_power : float, array_like
            Spherically recovered power.  The major index corresponds to
            spherical degrees and the minor index to matching spherical
            orders.

        """
        density_contrast_coeff = self.density_constrast(method=method)

        spherical_power = self._square_amplitude(
            density_contrast_coeff,
            normalisation=self.disc.normalisations
        )

        return spherical_power

    def two_points_pivoted(self, pivot, method=None, order_collapse=False):
        r"""Comptute 2-point values given a pivot for unpacking indices to
        a flat vector.

        Parameters
        ----------
        pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Pivot axis order for unpacking indices.
        method : str, optional
            Expectation computation method (default is `None`) if the
            transformed spherical Fourier coefficients have not been
            computed.
        order_collapse : bool, optional
            If `True` (default is `False`), spherical Fourier coefficients
            are first collapsed over spherical orders.

        Returns
        -------
        complex :class:`numpy.ndarray`
            2-point values as 2-d array.

        See Also
        --------
        :class:`~.morph.SphericalArray` : spherical array structure.

        """
        density_contrast_coeff = self.density_constrast(method=method)

        return self._compute_pivoted_two_points_from_coeff(
            density_contrast_coeff, self.disc,
            pivot=pivot,
            order_collapse=order_collapse
        )

    @staticmethod
    def _square_amplitude(density_contrast_coeff, normalisation=None):
        """Compute normalised square amplitudes from spherical Fourier
        coefficients of the field.

        Parameters
        ----------
        density_contrast_coeff : *dict of* {*int*: *complex* |ndarray|}
            Spherical Fourier density contrast coefficients.
        normalisation : *dict of* {*int*: |ndarray|}, optional
            Normalisation coefficients.  If `None`, all normalisation
            coefficients are set to unity.

        Returns
        -------
        float, array_like
            Spherically recovered power with given normalisation.


        .. |ndarray| replace:: :class:`numpy.ndarray`

        """
        sorted_degrees = np.sort(list(density_contrast_coeff.keys()))

        if normalisation is None:
            normalisation = dict.fromkeys(sorted_degrees, 1)

        return [
            normalisation[ell] * np.average(
                np.abs(density_contrast_coeff[ell])**2, axis=0
            )
            for ell in sorted_degrees
        ]

    @staticmethod
    def _compute_pivoted_two_points_from_coeff(density_contrast_coeff, disc,
                                               pivot, order_collapse=False):
        r"""Compute 2-point values from spherical Fourier coefficients.

        Parameters
        ----------
        density_contrast_coeff : *dict of* {*int*: *complex* |ndarray|}
            Spherical Fourier density contrast coefficients.
        disc : :class:`~.mapper.spherical_transform.DiscreteSpectrum`
            Discrete spectrum.
        pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
            Pivot axis order for unpacking indices.
        order_collapse : bool, optional
            If `True` (default is `False`), coefficients `n_coeff`,
            `nbar_coeff` are first collapsed over spherical orders.

        Returns
        -------
        complex :class:`numpy.ndarray`
            2-point values as 2-d array.

        See Also
        --------
        :class:`~.morph.SphericalArray` : spherical array structure.


        .. |ndarray| replace:: :class:`numpy.ndarray`

        """
        delta_ellmn = SphericalArray.build(
            disc=disc, filling=density_contrast_coeff
        )

        if order_collapse:
            collapse = 'rms'
        else:
            collapse = None

        delta_ellmn_flat = delta_ellmn.unfold(
            pivot, collapse=collapse, return_only='data'
        )

        return np.outer(delta_ellmn_flat, np.conj(delta_ellmn_flat))
