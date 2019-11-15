"""
Window function (:mod:`~harmonia.algorithms.window`)
===========================================================================

Determine the window function from survey specifications with high-density
synthetic catalogues.

.. autosummary::

    WindowFunction

|

"""
import mcfit
import numpy as np
from nbodykit.lab import ConvolvedFFTPower, UniformCatalog, FKPCatalog
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from harmonia.collections.utils import cartesian_to_spherical as c2s


class WindowFunction:
    """Window function in configuration space determined from synthetic
    catalogues.

    Parameters
    ----------
    number_density : float
        Particle number density (in cubic h/Mpc).
    boxsize : float, array_like
        Catalogue box size (in Mpc/h) as a scalar or a triple of scalars.
    mask, selection, weight : callable or None, optional
        Survey specification functions (default is `None`).
    source : {'simulation', 'survey'}, optional
        Source type of the catalogue.  If ``'simulation'`` (default),
        `mask`, `selection` and `weight` are assumed to be functions of
        Cartesian coordinates whose origin is the centre of the catalogue;
        if ``'survey'``, `mask`, `selection` and `weight` are assumed to be
        functions of spherical coordinates.

    Attributes
    ----------
    mask, selection, weight : callable or None
        Survey specification functions.
    source : {'simulation', 'survey'}
        Source type of the catalogue.  If ``'simulation'``, `mask`,
        `selection` and `weight` are assumed to be functions of Cartesian
        coordinates whose origin is the centre of the catalogue; if
        ``'survey'``, `mask`, `selection` and `weight` are assumed to be
        functions of spherical coordinates.
    synthetic_catalogue : :class:`nbodykit.base.catalog.CatalogSource`
        Synthetic catalogue with the specified window.

    """

    def __init__(self, number_density, boxsize, mask=None, selection=None,
                 weight=None, source='simulation', padding=80):

        self.source, self.mask, self.selection, self.weight = \
            source, mask, selection, weight

        self.synthetic_catalogue = self._synthethise(
            number_density,
            boxsize,
            self.source,
            self.mask,
            self.selection,
            self.weight,
            padding=padding
        )

    def power_spectrum_multipoles(self, degrees, kmin=0., kmax=None, dk=None,
                                  **mesh_kwargs):
        """Determine window function power spectrum multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        degrees : int, array_like
            Multipole degrees.
        kmin, kmax, dk : float or None, optional
            Maximum wavenumber, minimum wavenumber or wavenumber bin size
            (default is `None`).
        mesh_kwargs, power_kwargs : dict or None, optional
            Keyword arguments to be passed to
            :class:`nbodykit.source.mesh.catalog.CatalogMesh` for FFT
            gridding of the synthetic catalogue or
            :class:`nbodykit.algorithms.ConvolvedFFTPower` for computing
            multipoles of the synthetic catalogue.

        Returns
        -------
        power_multipoles : :class:`nbodykit.binned_statistic.BinnedStatistic`
            Binned multipole statistics.

        """
        if 'Nmesh' not in mesh_kwargs:
            mesh_kwargs['Nmesh'] = 256
        if 'resampler' not in mesh_kwargs:
            mesh_kwargs['resampler'] = 'tsc'
        if 'compensated' not in mesh_kwargs:
            mesh_kwargs['resampler'] = True
        if kmax is None:
            kmax = np.pi * mesh_kwargs['Nmesh'] \
                / self.synthetic_catalogue.attrs['BoxSize']
        if dk is None:
            dk = 2*2*np.pi / self.synthetic_catalogue.attrs['BoxSize']

        power_multipoles = ConvolvedFFTPower(
            self.synthetic_catalogue.to_mesh(**mesh_kwargs),
            degrees,
            kmin=kmin,
            kmax=kmax,
            dk=dk
        ).poles

        self.power_multipoles = power_multipoles

        return power_multipoles

    def correlation_function_multipoles(self, degrees, **power_kwargs):
        """Determine window function correlation function multipoles from the
        synthetic catalogue.

        Parameters
        ----------
        degrees : int, array_like
            Multipole degrees.
        **power_kwargs
            Keyword arguments to be passed to
            :class:`~WindowFunction.power_spectrum_multipoles` for
            computing multipoles of the synthetic catalogue.

        Returns
        -------
        poles : :class:`nbodykit.binned_statistic.BinnedStatistic`
            Binned multipole statistics.

        """
        max_required_degree = max(degrees) + 4
        required_degrees = list(range(0, max_required_degree+1, 2))

        try:
            power_multipoles = self.power_multipoles
            assert hasattr(
                power_multipoles, 'power_{:d}'.format(max_required_degree)
            )
        except (AttributeError, AssertionError):
            power_multipoles = self.power_spectrum_multipoles(
                required_degrees, **power_kwargs
            )

        # waves = power_multipoles['modes']
        poles = {ell: power_multipoles[f'power_{ell}'] for ell in degrees}

        return poles

    @staticmethod
    def _synthethise(number_density, boxsize, source, mask, selection, weight,
                     padding):

        catalogue = UniformCatalog(number_density, boxsize)

        catalogue['Location'] = catalogue['Position'] - np.divide(boxsize, 2)

        if source == 'simulation':
            pass
        elif source == 'survey':
            catalogue['Location'] = c2s(catalogue['Location'])
        else:
            raise ValueError("Unrecognised source type: '{}'. ".format(source))

        if callable(mask):
            catalogue['Weight'] *= mask(catalogue['Location'])
        if callable(selection):
            catalogue['Weight'] *= selection(catalogue['Location'])
        if callable(weight):
            catalogue['Weight'] *= weight(catalogue['Location'])

        catalogue['NZ'] = number_density * catalogue['Weight']

        synthetic_catalogue = FKPCatalog(
            catalogue, None, BoxSize=padding*boxsize
        )

        return synthetic_catalogue
