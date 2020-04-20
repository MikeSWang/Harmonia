"""
Map transform (:mod:`~harmonia.mapper.map_transform`)
===========================================================================

Transform discrete catalogues to Fourier-space map data.

.. autosummary::

    SphericalMap
    CartesianMap

|

"""
import logging

import numpy as np
from nbodykit.lab import ConvolvedFFTPower

from harmonia.algorithms.discretisation import DiscreteSpectrum
from harmonia.algorithms.arrays import CartesianArray, SphericalArray
from harmonia.algorithms.bases import spherical_besselj, spherical_harmonic
from harmonia.surveyor.coordinates import cartesian_to_spherical as c2s
from harmonia.utils import Progress


class SphericalMap:
    r"""Discretised spherical Fourier map from catalogue sources.

    Parameters
    ----------
    catalogues : :class:`~.catalogue_maker.SphericalFKPCatalogue`
        FKP-style paired catalogues in a spherical domain.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum for the map.
    initialise : bool, optional
        If `True` (default), map transform is performed upon creation.

    Attributes
    ----------
    catalogues : :class:`~.catalogue_maker.SphericalFKPCatalogue`
        FKP-style paired catalogues associated with the map.
    disc : :class:`~harmonia.algorithms.discretisation.DiscreteSpectrum`
        Discrete spectrum associated with the map.
    attrs : dict
        Attributes inherited upon creation.

    """

    def __init__(self, catalogues, disc, initialise=True):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.catalogues = catalogues
        self.disc = disc

        self.attrs = {}
        self.attrs.update(self.catalogues.attrs)
        self.attrs.update(self.disc.attrs)

        # Coefficient directories which internally store the
        # transformed results.
        self._data_coeff = {}
        self._rand_coeff = {}

        self._density_contrast = None
        self._mode_power = None

        if initialise:
            self._density_contrast = self._transform()

    def __str__(self):

        str_info = "{}, source={}, contrast={}".format(
            self.disc, self.attrs['source'], self.attrs['contrast']
        )

        return f"{self.__class__.__name__}({str_info})"

    def __setstate__(self, state):

        for attr, val in state.items():
            if attr == 'disc':
                self.disc = DiscreteSpectrum._from_state(state['disc'])
            elif attr == 'density_contrast':
                self.density_contrast = \
                    SphericalArray._from_state(state['density_contrast'])
            elif attr.endswith('_coeff'):
                setattr(self, '_' + attr, val)
            else:
                setattr(self, attr, val)

    def __getstate__(self):

        state = {
            'attrs': self.attrs,
            'disc': self.disc.__getstate__(),
            'data_coeff': self._data_coeff,
            'rand_coeff': self._rand_coeff,
            'density_contrast': self.density_contrast.__getstate__(),
            'mode_power': self.mode_power,
        }

        return state

    @classmethod
    def _from_state(cls, state):

        self = object.__new__(cls)

        self.__setstate__(state)

        # pylint: disable=protected-access
        if not hasattr(self, '_mode_power'):
            self._mode_power = None
        if not hasattr(self, '_data_coeff'):
            self._data_coeff = {}
        if not hasattr(self, '_rand_coeff'):
            self._rand_coeff = {}

        return self

    @property
    def density_contrast(self):
        """Spherical Fourier coefficients for the density contrast between
        data and random catalogues.

        Notes
        -----
        When the spherical map is initialsed upon creation or this
        is directly or indirectly accessed without initialising the
        spherical map upon creation, discrete spherical Fourier transform
        is performed by direct summation over all selected objects in the
        paired catalogues.

        Calling this method stores the transformed number densities of
        data and random catalogues internally for further processing,
        e.g. for :meth:`mode_power`.  Computational redundancy is reduced
        by employing parity relations between spherical harmonics of the
        same degree but opposite orders.

        Returns
        -------
        :class:`~harmonia.algorithms.arrays.SphericalArray`
            Density contrast coefficients of the spherical map.

        """
        if self._density_contrast is not None:
            return self._density_contrast

        self._density_contrast = self._transform()

        return self._density_contrast

    @density_contrast.setter
    def density_contrast(self, value):

        if isinstance(value, SphericalArray):
            self._density_contrast = value
        else:
            raise TypeError(
                "Only SphericalArray objects can be set as `density_contrast`."
            )

    @property
    def mode_power(self):
        """Spherical Fourier mode power suitably normalised.

        In the simplest case of a full-sky statistically isotropic map,
        this is equivalent to the power spectrum at the same mode
        wavenumbers.

        Returns
        -------
        dict
            Spherical mode power at discrete mode wavenumbers with mode
            counts.

        """
        if self._mode_power is not None:
            return self._mode_power

        # Sort all variables by wavenumber.
        sort_order = np.argsort(
            list(self.disc.wavenumbers.values())
        )

        wavenumbers = np.array(
            list(self.disc.wavenumbers.values())
        )[sort_order]

        normalisations = np.array(
            list(self.disc.normalisations.values())
        )[sort_order]

        mode_counts = np.zeros_like(wavenumbers)
        mode_powers = np.zeros_like(wavenumbers)

        # Filter the spherical structure array by wavenumber and flatten.
        for idx, k in enumerate(wavenumbers):
            selector = (self.density_contrast.array['wavenumber'] == k)
            subarray = self.density_contrast.array['coefficient'][selector]
            mode_counts[idx] = np.sum(selector)
            mode_powers[idx] = \
                normalisations[idx] * np.mean(np.abs(subarray) ** 2)

        return {
            'wavenumbers': wavenumbers,
            'mode_counts': mode_counts,
            'mode_powers': mode_powers,
        }

    def _transform(self):

        density_contrast = SphericalArray(self.disc)

        if not self._data_coeff or not self._rand_coeff:  # if either is empty
            progress = Progress(
                density_contrast.size,
                process_name="spherical transform",
                logger=self.logger
            )

            self.logger.info("Transforming %s...", self)

            # Transform degree by degree and update the coefficient directory.
            for deg_idx, deg in enumerate(self.disc.degrees):
                self._transform_degree(deg)
                progress.report(sum(self.disc.mode_counts[:(deg_idx + 1)]) - 1)

            self.logger.info("... %s transformed.", self)

        self.logger.info("Computing density contrast for %s...", self)

        # Subtract random catalogue coefficients and normalise by data
        # catalogue mean number density to obtain density contrast.
        for mode_pos, mode_index in enumerate(density_contrast.array['index']):
            density_contrast[mode_pos] = (
                self._data_coeff[tuple(mode_index)]
                - self._rand_coeff[tuple(mode_index)]
            ) / self.catalogues.data_catalogue.attrs['nbar']

        self.logger.info("... computed density contrast for %s.", self)

        return density_contrast

    def _transform_degree(self, ell):

        nmax = self.disc.depths[self.disc.degrees.index(ell)]

        loc_data = c2s(self.catalogues.data_catalogue['Location'])
        loc_rand = c2s(self.catalogues.random_catalogue['Location'])
        vet_data = self.catalogues.data_catalogue['Selection']
        vet_rand = self.catalogues.random_catalogue['Selection']
        sel_data = self.catalogues.data_catalogue['NZ'] \
            / self.catalogues.data_catalogue.attrs['nbar']
        sel_rand = self.catalogues.random_catalogue['NZ'] \
            / self.catalogues.data_catalogue.attrs['nbar']
        wgt_data = self.catalogues.data_catalogue['Weight']
        wgt_rand = self.catalogues.random_catalogue['Weight']

        coeff_data, coeff_rand = {}, {}

        for m in range(- ell, 1):
            for n in range(1, nmax + 1):
                k = self.disc.wavenumbers[(ell, n)]

                # Unpack spherical coordinates.
                r_data, t_data, p_data = loc_data.swapaxes(0, 1)[:]
                r_rand, t_rand, p_rand = loc_rand.swapaxes(0, 1)[:]

                coeff_data[(ell, m, n)] = complex(
                    np.sum(
                        vet_data[:] * sel_data[:] * wgt_data[:]
                        * spherical_besselj(ell, k * r_data)
                        * spherical_harmonic(ell, m, t_data, p_data, conj=True)
                    )
                )
                coeff_rand[(ell, m, n)] = complex(
                    np.sum(
                        vet_rand[:] * sel_rand[:] * wgt_rand[:]
                        * spherical_besselj(ell, k * r_rand)
                        * spherical_harmonic(ell, m, t_rand, p_rand, conj=True)
                    ) / self.catalogues.attrs['contrast']
                )

        # Employ parity relations for positive spherical orders.
        for m in range(1, ell + 1):
            for n in range(1, nmax + 1):
                coeff_data[(ell, m, n)] = (-1) ** m * np.conj(
                    coeff_data[(ell, - m, n)]
                )
                coeff_rand[(ell, m, n)] = (-1) ** m * np.conj(
                    coeff_rand[(ell, - m, n)]
                )

        self._data_coeff.update(coeff_data)
        self._rand_coeff.update(coeff_rand)


class CartesianMap:
    """Compressed Cartesian Fourier-space map from catalogue sources.

    Parameters
    ----------
    catalogues : :class:`~.SphericalFKPCatalogue` or :class:`FKPCatalog`
        FKP-style paired catalogues in a spherical domain.
    orders : sequence of int
        Orders of the power spectrum multipoles.
    kmin : float, optional
        Minimum wavenumber of the compressed map (in :math:`h`/Mpc)
        (default is `None`).
    kmax : float or None, optional
        Maximum wavenumber of the compressed map (in :math:`h`/Mpc)
        (default is `None`).
    dk : float or None, optional
        Wavenumber bin width (in :math:`h`/Mpc) (default is `None`).
    num_mesh : int, optional
        Mesh number per dimension for interpolating the discrete
        catalogues on a grid.
    resampler : {'cic', 'tsc', 'pcs'}, optional
        Grid assignment scheme (default is 'tsc') for catalogue
        interpolation.
    interlaced : bool, optional
        If `True` (default), use interlacing for aliasing mitigation.

    Attributes
    ----------
    catalogues : :class:`~.catalogue_maker.SphericalFKPCatalogue`
        FKP-style paired catalogues associated with the map.
    mesh : :class:`nbodykit.base.mesh.MeshSource`
        FFT mesh of the interpolated catalogues.
    power_multipoles : :class:`~harmonia.algorithms.arrays.CartesianArray`
        Power spectrum multipoles of different Legendre orders for
        wavenumber bins.

    """

    def __init__(self, catalogues, orders, kmin=None, kmax=None, dk=None,
                 num_mesh=256, resampler='tsc', interlaced=True):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.catalogues = catalogues

        self.attrs = {arg: val for arg, val in locals().items() if arg != 'dk'}
        self.attrs.update(catalogues.attrs)

        mesh_kwargs = dict(
            Nmesh=num_mesh, resampler=resampler, interlaced=interlaced,
            compensated=True
        )
        try:  # deal with `SphericalFKPCatalogue`
            self.mesh = catalogues.catalogue_pair.to_mesh(**mesh_kwargs)
        except AttributeError:  # deal with `nbodykit`'s `FKPCatalog`
            self.mesh = catalogues.to_mesh(**mesh_kwargs)

        self.logger.debug("Mesh painted on a grid for %s.", catalogues)

        kmin = kmin or 1.e-4

        self.power_multipoles = self._compress(orders, kmin, kmax, dk)

    def __str__(self):

        str_info = "source={}, orders={}, scales=[{},{}], contrast={}".format(
            self.attrs['source'], self.attrs['orders'],
            self.attrs['kmin'], self.attrs['kmax'],
            self.attrs['contrast']
        )

        return f"{self.__class__.__name__}({str_info})"

    def __setstate__(self, state):

        for attr, val in state.items():
            setattr(self, attr, val)

    def __getstate__(self):

        state = {
            'attrs': self.attrs,
            'power_multipoles': self.power_multipoles,
        }

        return state

    @classmethod
    def _from_state(cls, state):  # internal classmethod

        self = object.__new__(cls)

        self.__setstate__(state)

        return self

    def _compress(self, orders, kmin, kmax, dk):

        results = ConvolvedFFTPower(
            self.mesh, poles=orders, kmin=kmin, kmax=kmax, dk=dk
        ).poles

        # Remove spurious leading bins from `nbodykit`.
        valid_bins = ~np.equal(results['modes'], 0) \
            & ~np.equal(results['modes'], 1)

        wavenumbers = results['k'][valid_bins]
        mode_counts = results['modes'][valid_bins]
        shot_noise = results.attrs['shotnoise']

        multipoles = CartesianArray(
            orders, wavenumbers,
            mode_counts=mode_counts, shot_noise=shot_noise
        )

        self.logger.info(
            "Compressed catalogues into power spectrum multipoles."
        )

        # Fill in Cartesian structure array.
        multipoles[:] = np.concatenate(
            [results[f'power_{order}'][valid_bins].real for order in orders]
        )

        return multipoles
