"""
Synthesis (:mod:`~harmonia.surveyor.synthesis`)
===========================================================================

Systhesise FKP-style paired random catalogues for given survey
specifications.

This can be used for determining survey window functions and estimating
correlation induced by geometric filtering.

.. autosummary::

    SyntheticCatalogue
    CovarianceEstimator
    generate_compression_matrix

|

"""
import warnings

import numpy as np
import psutil
from mcfit import P2xi
from nbodykit.lab import FKPCatalog
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.algorithms.arrays import CartesianArray
from harmonia.mapper.catalogue_maker import (
    RandomCatalogue,
    SphericalFKPCatalogue,
)
from harmonia.mapper.map_transform import CartesianMap, SphericalMap
from harmonia.reader.likelihoods import spherical_covariance


class SyntheticCatalogue:
    r"""Synthetic catalogue for probing survey geometry.

    Parameters
    ----------
    mean_density : float
        Mean number density (in cubic :math:`h`/Mpc).
    boxsize : float, array_like
        Catalogue boxsize (in Mpc/:math:`h`) as a scalar or a triple of
        scalars.
    contrast : float or None, optional
        Contrast of the number density of the secondary random catalogue
        to that of the primary.  If `None` (default), the secondary
        catalogue is not produced.  This cannot be `None` if
        `sphericalise` is provided.
    expansion : float, optional
        Expansion factor of the catalogue box for grid assignment
        (default is 1.).  Ignored if `sphericalise` is set.
    sphericalise : float or None, optional
        If not `None` (default), this is passed to the `radius` parameter
        of :class:`~.mapper.catalogue_maker.SphericalFKPCatalogue` for
        instantiating the synthetic catalogue.
    mask, selection, weight : callable or None, optional
        Survey mask, selection or weight function (default is `None`).
        Must be given as a function of Cartesian coordinate only assuming
        the origin is at the centre of the catalogues.
    apply_selection_as_veto : bool, optional
        If `True` (default is `False`), `selection` returning boolean
        values is applied to the 'Selection' column of the catalogues
        instead of the 'NZ' column (see `nbodykit` documentation for
        this peculiarity.)

    Attributes
    ----------
    spherical_coefficients : :class:`~.arrays.SphericalArray` or None
        Density contrast coefficients.  Only available after
        :meth:`compute_spherical_coefficients` is called.
    power_multipoles : :class:`~.algorithms.arrays.CartesianArray` or None
        Power spectrum multipoles.  Only available after
        :meth:`compute_power` is called.
    window_multipoles : :class:`~.algorithms.arrays.CartesianArray` or None
        Power spectrum multipoles of the window function at sampled
        separations.  Only available after :meth:`compute_window` is
        called.
    mask_multipoles : :class:`~.algorithms.arrays.CartesianArray` or None
        Correlation function multipoles of the window function at sampled
        separations.  Only available after :meth:`compute_mask` is
        called.
    attrs : dict
        Attributes inherited from input parameters.

    """

    def __init__(self, number_density, boxsize, contrast=None, expansion=1.,
                 sphericalise=None, mask=None, selection=None, weight=None,
                 apply_selection_as_veto=False):

        boxsize = [boxsize,] * 3 if np.isscalar(boxsize) \
            else boxsize

        self.attrs = {
            'number_density': number_density,
            'boxsize': boxsize,
            'contrast': contrast,
            'expansion': expansion,
            'sphericalise': sphericalise,
        }

        self._synthetic_catalogue = self._synthesise(
            number_density, boxsize, contrast, expansion,
            mask, selection, weight, apply_selection_as_veto,
            radius=sphericalise
        )

        self.spherical_coefficients = None
        self.power_multipoles = None
        self.window_multipoles = None
        self.mask_multipoles = None

        self._intrpl_k = None
        self._intrpl_power = None
        self._intrpl_s = None
        self._intrpl_correlation = None

    def __str__(self):

        str_info = ", ".join(
            [f"{name}={val}" for name, val in self.attrs.items()]
        )

        return f"{self.__class__.__name__}({str_info})"

    def compute_spherical_coefficients(self, disc):
        """Compute the spherical Fourier density contrast coefficients of
        the synthetic catalogue.

        Parameters
        ----------
        disc : :class:`~.algorithms.discretisation.DiscreteSpectrum`
            Discrete spectrum.

        Returns
        -------
        :class:`~harmonia.algorithms.arrays.SphericalArray`
            Density contrast coefficients.

        See Also
        --------
        :class:`~harmonia.mapper.map_transform.SphericalMap`

        """
        spherical_map = SphericalMap(self._synthetic_catalogue, disc)

        return spherical_map.density_contrast

    def compute_power(self, orders, kmin=1.e-4, kmax=None, dk=None,
                      num_mesh=256, resampler='tsc', interlaced=True):
        """Compute the power spectrum multipoles of the synthetic
        catalogue.

        Parameters
        ----------
        orders : sequence of int
            Orders of the power spectrum multipoles.
        kmin : float, optional
            Minimum wavenumber of the map (in :math:`h`/Mpc)
            (default is 1.e-4).
        kmax : float or None, optional
            Maximum wavenumber of the map (in :math:`h`/Mpc)
            (default is `None`).  If `None`, `kmax` is the Nyquist
            wavenumber determined from `num_mesh`.
        dk : float or None, optional
            Wavenumber bin width (in :math:`h`/Mpc) (default is `None`).
        num_mesh : int, optional
            Mesh number per dimension for interpolating the discrete
            catalogues on a grid.
        resampler : {'cic', 'tsc', 'pcs'}, optional
            Grid assignment scheme (default is ``'tsc'``) for catalogue
            interpolation.
        interlaced : bool, optional
            If `True` (default), use interlacing for aliasing mitigation.

        Returns
        -------
        :class:`~harmonia.algorithms.arrays.CartesianArray`
            Power spectrum multipoles binned in wavenumbers.

        See Also
        --------
        :class:`~harmonia.mapper.map_transform.CartesianMap`

        """
        if self.power_multipoles is not None:
            warnings.warn(
                "Power spectrum multipoles have already been computed; "
                "they are now being recomputed."
            )

        # Check if there is sufficient memory; if not,
        # reduce the mesh number.
        while psutil.virtual_memory().available < 10 * 768**3:
            num_mesh /= 2
            warnings.warn(
                "Mesh number reduced due to limited memory", RuntimeWarning
            )

        # Set the Nyquist wavenumber as ther upper limit.
        if kmax is None:
            kmax = np.pi * num_mesh \
                / min(self._synthetic_catalogue.attrs['BoxSize'])

        cartesian_map = CartesianMap(
            self._synthetic_catalogue,
            orders, kmin=kmin, kmax=kmax, dk=dk,
            num_mesh=num_mesh, resampler=resampler, interlaced=interlaced
        )

        with warnings.catch_warnings():
            # Filter out warnings when the secondary catalogue
            # in the synthetic FKP-style catalogue pair is None,
            # as it would cause division by zero.
            warnings.filterwarnings(
                'ignore',
                category=RuntimeWarning,
                message=".*invalid value encountered.*"
            )
            warnings.filterwarnings(
                'ignore',
                category=RuntimeWarning,
                message="divide by zero encountered in double_scalars"
            )
            self.power_multipoles = cartesian_map.power_multipoles

        return self.power_multipoles

    def compute_window(self, orders, kmin=1.e-6, **kwargs):
        """Compute the Fourier-space window function multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        orders : sequence of int
            Orders of the power spectrum multipoles.
        kmin : float, optional
            Minimum wavenumber (in :math:`h`/Mpc) (default is 1.e-6).
        **kwargs
            Any other parameters to be passed to :meth:`~.compute_power`.

        Returns
        -------
        :class:`~harmonia.algorithms.arrays.CartesianArray`
            Window function multipoles binned in logarithmic wavenumbers.

        See Also
        --------
        :meth:`~.compute_power`

        """
        LOG10_K_MAX = 1.
        NUM_EXTENSION = 1000
        NUM_INTERPOLATION = pow(2, 10)

        if np.isclose(self.attrs['expansion'], 1.):
            warnings.warn(
                "No padding for catalogue box when computing "
                "the window function.  You may want to resynthesise "
                "the catalogue with `expansion` set above 1."
            )

        # Need to compute power spectrum multipoles first.
        power_multipoles = self.compute_power(orders, kmin=kmin, **kwargs)

        raw_k = power_multipoles.attrs['wavenumbers']
        raw_power = {
            ell: power_multipoles.array['power'][
                power_multipoles.array['order'] == ell
            ]
            for ell in orders
        }

        # Linear padding and then logarithmic padding in wavenumbers.
        padding = np.mean(np.abs(np.diff(raw_k)))
        lin_extension = np.max(raw_k) + padding * np.arange(1, NUM_EXTENSION)
        log_extension = np.logspace(
            np.log10(np.max(raw_k) + padding * NUM_EXTENSION),
            LOG10_K_MAX,
            num=NUM_EXTENSION
        )

        ext_k = np.r_[raw_k, lin_extension, log_extension]
        ext_power = {
            ell: np.r_[raw_power[ell], np.zeros(2 * NUM_EXTENSION - 1)]
            for ell in orders
        }

        # Resample at log-spaced wavenumbers.
        self._intrpl_k = np.logspace(
            *np.log10(ext_k[[0, -1]]), num=NUM_INTERPOLATION
        )
        self._intrpl_power = {
            ell: Spline(ext_k, ext_power[ell], k=1)(self._intrpl_k)
            for ell in orders
        }

        # Store as an attribute.
        orders = np.sort(orders).tolist()

        self.window_multipoles = CartesianArray(orders, self._intrpl_k)
        self.window_multipoles[:] = np.concatenate(
            [self._intrpl_power[ell] for ell in orders]
        )

        # Normalise all values to the monopole at vanishing wavenumbers.
        self.window_multipoles[:] /= self.window_multipoles[0]

        return self.window_multipoles

    def compute_mask(self, orders, *args, **kwargs):
        """Compute the configuration-space mask function multipoles from
        the synthetic catalogue by Hankel transform of the window function
        multipoles.

        Parameters
        ----------
        orders : sequence of int
            Orders of the mask function multipoles.
        *args, **kwargs
            Any other position and keyword parameters to be passed to
            :meth:`~.compute_window`.  Only required when
            :attr:`window_multipoles` is not available.

        Returns
        -------
        :class:`numpy.ndarray`
            Mask function multipoles binned in logarithmic separations.
            This is a NumPy structured array similar to
            :attr:`array` of :class:`~.algorithms.arrays.CartesianArray`.

        See Also
        --------
        :class:`~.algorithms.arrays.CartesianArray`

        """
        NUM_INTERPOL = pow(2, 14)

        # Need to compute window function multiples first.
        if self.window_multipoles is None \
                or max(orders) not in self._intrpl_power:
            self.window_multipoles = self.compute_window(
                orders, *args, **kwargs
            )

        # Hankel transform window function multipoles.
        with warnings.catch_warnings():  # suppress warnings from `mcfit`
            warnings.filterwarnings(
                'ignore',
                message=(
                    "The default value of extrap has been changed to False, "
                    "set it to True if you cannot reproduce previous results"
                )
            )
            raw_correlation = {
                ell: P2xi(self._intrpl_k, l=ell, lowring=True)(
                    self._intrpl_power[ell], extrap=False
                )
                for ell in orders
            }

        raw_s = [raw_correlation[ell][0] for ell in orders]
        raw_correlation.update({ell: raw_correlation[ell] for ell in orders})

        # Resample at log-spaced separations.
        self._intrpl_s = np.logspace(
            np.log10(np.min(raw_s)), np.log10(np.max(raw_s)), num=NUM_INTERPOL
        )
        self._intrpl_correlation = {
            ell: Spline(*raw_correlation[ell], k=1)(self._intrpl_s)
            for ell in orders
        }

        # Create a structure array dedicated for storing mask multipoles.
        self.mask_multipoles = np.empty(
            np.size(orders) * np.size(self._intrpl_s),
            dtype=np.dtype({
                'names': ['order', 'separation', 'correlation'],
                'formats': ['i4', 'f8', 'f8'],
            })
        )

        self.mask_multipoles['order'] = np.repeat(orders, len(self._intrpl_s))
        self.mask_multipoles['separation'] = np.tile(
            self._intrpl_s, len(orders)
        )

        # Normalise all values to the monopole at vanishing separations.
        self.mask_multipoles['correlation'] = np.concatenate(
            [self._intrpl_correlation[ell] for ell in orders]
        ) / raw_correlation[0][-1][0]

        return self.mask_multipoles

    @staticmethod
    def _synthesise(number_density, boxsize, contrast, expansion,
                    mask, selection, weight, apply_selection_as_veto,
                    radius=None):

        primary_catalogue = RandomCatalogue(number_density, boxsize)

        if contrast is not None:
            secondary_catalogue = RandomCatalogue(
                contrast * number_density, boxsize
            )
        else:
            secondary_catalogue = None

        # No sphericalisation; reimplement FKP build similar to
        # :class:`~harmonia.mapper.catalogue_maker.SphericalFKPCatalogue`.
        if radius is None:
            for catalogue in [primary_catalogue, secondary_catalogue]:
                if catalogue is None:
                    break

                catalogue['Location'] = \
                    catalogue['Position'] - np.divide(boxsize, 2)
                catalogue['NZ'] = number_density * catalogue['Selection']

                if callable(mask):
                    catalogue['Selection'] *= mask(catalogue['Location'])
                if callable(selection):
                    if apply_selection_as_veto:
                        catalogue['Selection'] *= \
                            selection(catalogue['Location'])
                    else:
                        catalogue['NZ'] *= selection(catalogue['Location'])
                if callable(weight):
                    catalogue['Weight'] *= weight(catalogue['Location'])

            return FKPCatalog(
                primary_catalogue, secondary_catalogue,
                BoxSize=np.multiply(expansion, boxsize)
            )

        # Reuse :class:`~.mapper.catalogue_maker.SphericalFKPCatalogue`.
        spherical_catalogue = SphericalFKPCatalogue(
            radius,
            data_catalogue=primary_catalogue,
            random_catalogue=secondary_catalogue,
            mask=mask, selection=selection, weight=weight,
            apply_selection_as_veto=apply_selection_as_veto
        )

        return FKPCatalog(
            spherical_catalogue.data_catalogue,
            spherical_catalogue.random_catalogue,
            BoxSize=np.multiply(expansion, boxsize)
        )


class CovarianceEstimator:
    """Covariance matrix estimator for power spectrum multipoles at
    fiducial values.

    Parameters
    ----------
    realisations : sequence of :class:`~.algorithms.arrays.CartesianArray`
        Independent realisations of power spectrum multipoles from which
        the covariance matrix is estimated.  For each realisation, orders
        and wavenumbers of the multipoles must be sorted.
    reference : :class:`~.arrays.CartesianArray` *or None, optional*
        Underlying power spectrum multipoles binned in wavenumber that
        are realised for covariance estimation.  Orders and wavenumbers of
        the multipoles must be sorted.  If `None` (default), this is
        determined by the average of `realisations`.

    Attributes
    ----------
    realisations : sequence of :class:`~.algorithms.arrays.CartesianArray`
        Independent realisations of power spectrum multipoles from which
        the covariance matrix is estimated.
    reference : :class:`~.arrays.CartesianArray` or None
        Underlying power spectrum multipoles binned in wavenumber that
        are realised for covariance estimation.
    wavenumbers : float :class:`ndarray`
        Wavenumbers at which the covariance matrix is estimated.
        This is set by `reference` if available; otherwise it is set by
        the first of `realisations`.

    See Also
    --------
    :class:`~harmonia.algorithms.arrays.CartesianArray`

    """

    def __init__(self, realisations, reference=None):

        self.realisations = realisations
        self.reference = reference

        if self.reference is not None:
            self.wavenumbers = np.unique(self.reference.array['wavenumber'])
        else:
            self.wavenumbers = np.unique(
                self.realisations[0].array['wavenumber']
            )

    def __setstate__(self, state):

        for attr, value in state.items():
            if attr == 'reference' and value is not None:
                reference = object.__new__(CartesianArray)
                reference.__setstate__(value)
                setattr(self, 'reference', reference)
            else:
                setattr(self, attr, value)

    def __getstate__(self):

        state = self.__dict__

        if self.reference is not None:
            state.update({'reference': self.reference.__getstate__()})

        return state

    def save(self, output_file):
        """Save the estimator with its attributes as a .npz file.

        Parameters
        ----------
        output_file : *str or* :class:`pathlib.Path`
            Output file path.

        """
        np.savez(output_file, **self.__getstate__())

    @classmethod
    def load(cls, input_file):
        """Load the estimator with its attributes from a .npz file.

        Parameters
        ----------
        input_file : *str or* :class:`pathlib.Path`
            Input file path.

        """
        state_data = np.load(input_file)

        state = {}
        for attr in state_data.files:
            try:
                state.update({attr: state_data[attr].item()})
            except (AttributeError, ValueError):
                state.update({attr: state_data[attr]})

        self = object.__new__(cls)

        self.__setstate__(state)

        return self

    def get_fiducial_vector(self, pivot):
        """Return the fiducial data vector for which the covariance
        matrix is estimated.

        Parameters
        ----------
        pivot : {'order', 'wavenumber'}
            Pivot order used for data vectorisation.

        Returns
        -------
        float :class:`numpy.ndarray`
            Fiducial vector.

        """
        data_matrix = [
            realisation.vectorise(pivot) for realisation in self.realisations
        ]

        data_mean = np.mean(data_matrix, axis=0)

        if self.reference is None:
            return data_mean

        fiducial_vector = self.reference.vectorise(pivot)

        if not np.allclose(data_mean, fiducial_vector, rtol=0.1):
            warnings.warn(
                "Realisation average deviates from `truth` "
                "by more than 10% for some data entries."
            )

        return fiducial_vector

    def get_fiducial_covariance(self, pivot):
        """Return the fiducial covariance matrix estimated from data
        realisations.

        Parameters
        ----------
        pivot : {'order', 'wavenumber'}
            Pivot order used for data vectorisation.

        Returns
        -------
        float :class:`numpy.ndarray`
            Estimate fiducialcovariance.

        """
        data_matrix = [
            realisation.vectorise(pivot) for realisation in self.realisations
        ]

        return np.cov(data_matrix, rowvar=False, ddof=1)


def generate_compression_matrix(fiducial_model_kwargs,
                                extremal_model_kwargs=None,
                                sensitivity_threshold=0.01, discard=None):
    r"""Generate a compression matrix for spherical modes.

    Notes
    -----
    Compression is achieved by discarding non-positive eigenvalue modes
    that are at least :math:`10^{-8}` times smaller than the largest and
    in addition any of the following means:

        * `discard` is passed to discard a number of low-eigenvalue modes;
        * `extremal_model_kwargs` is passed and eigenvalues of the
          resulting model covariance are compared with those from
          `fiducial_covariance`.  Modes corresponding to low, insensitive
          (i.e. relative difference less than `sensitivity_threshold`)
          are discarded.
        * A combination of the above if the appropriate parameters
          are passed.

    Parameters
    ----------
    fiducial_model_kwargs : dict
        Fiducial model parameters to be passed to
        :func:`~.reader.likelihoods.spherical_covariance`.
    extremal_model_kwargs : dict or None, optional
        Extremal model parameters to be passed to
        :func:`~.reader.likelihoods.spherical_covariance`.
    sensitivity_threshold: float, optional
        Sensitivity threshold for modes deemed discardable
        (default is 0.01).
    discard : int or None, optional
        Number of low-eigenvalue modes to discard from all modes
        (default is `None`).

    Returns
    -------
    compression_matrix : :class:`numpy.ndarray`
        Compression matrix.

    """
    fiducial_covariance = spherical_covariance(**fiducial_model_kwargs)

    evals_fiducial, evecs = np.linalg.eigh(fiducial_covariance)

    selectors = []

    # Compression by positive magnitude.
    selectors.append(evals_fiducial > 1.e-8 * np.max(evals_fiducial))

    # Compression by discard.
    if discard is not None:
        selectors.append(np.arange(len(evals_fiducial)) >= discard)

    # Compression by comparison for sensitivity.
    extremal_covariance = spherical_covariance(**extremal_model_kwargs)

    evals_extremal = np.linalg.eigvalsh(extremal_covariance)

    selectors.append(
        ~np.isclose(evals_extremal, evals_fiducial, rtol=sensitivity_threshold)
    )

    # pylint: disable=no-member
    # Compress and reverse order.
    evecs = evecs[:, np.logical_and.reduce(selectors)][:, ::-1]

    compression_matrix = np.conj(evecs).T

    return compression_matrix
