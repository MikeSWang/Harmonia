"""
Gaussian random field (:mod:`~harmonia.algorithms.gaussian_random_field`)
===============================================================================

Make Gaussian random fields with matching power spectrum.

"""
import numpy as np
from scipy import fftpack as fftp


def gen_regular_grid(unitscale, ncells, retnorm=False):
    """Generate the coordinate grid for a cubic cell array of given cell number
    and length per dimension.

    Parameters
    ----------
    unitscale : float
        Length of each cell per dimension.
    ncells : int
        Number of cells per dimension.
    retnorm : bool, optional
        If `True` (default is `False`), also return the coordinate vector norm
        array for the grid.

    Returns
    -------
    coords_list : list [of length 3] of :class:`numpy.ndarray` of float
        Coordinate arrays for each dimension of the regular grid.
    normgrid : :class:`numpy.ndarray` of float, optional
        Coordinate vector norm array for the grid.  Returned if `retnorm` is
        `True`.

    """
    indxarr = np.indices((ncells,)*3)
    centres = np.array([(ncells - 1)/2]*3)

    coords_list = [
        unitscale * (indx - centre)
        for indx, centre in zip(indxarr, centres)
        ]

    if not retnorm:
        return coords_list

    normgrid = np.sqrt(sum([coords**2 for coords in coords_list]))

    return coords_list, normgrid


def gen_complex_white_noise(boxside, ncells, seed=None):
    """Generate a cubic array of random samples drawn from the standard complex
    normal distribution.

    Parameters
    ----------
    boxside : float
        Side length of the box.
    ncells : int
        Number of cells per dimension.
    seed : int or None, optional
        Global random seed (default is `None`).

    Returns
    -------
    whitenoise : :class:`numpy.ndarray` of complex
        Square array of random samples.

    """
    if seed is not None:
        np.random.seed(seed)

    whitenoise_real = np.random.normal(size=(ncells,)*3)
    whitenoise_imag = np.random.normal(size=(ncells,)*3)

    whitenoise = (whitenoise_real + 1j*whitenoise_imag) / np.sqrt(2)

    return whitenoise


def gen_gaussian_random_field(boxside, ncells, power_spectrum, seed=None):
    """Generate a Gaussian random field in a cubic box matching a given power
    spectrum.

    Parameters
    ----------
    boxside : float
        Side length of the Cartesian box (in Mpc/h).
    power_spectrum : callable
        Input power spectrum (in cubic Mpc/h).
    seed : int or None, optional
        Global random seed for the field (default is `None`).

    Returns
    -------
    field : :class:`numpy.ndarray` of float
        Gaussian random field in configuration space.

    """
    whitenoise = gen_complex_white_noise(boxside, ncells, seed=seed)

    volcell = (boxside/ncells)**3
    _, karr = gen_regular_grid(2*np.pi/boxside, ncells, retnorm=True)

    fourier_field = np.sqrt(volcell*power_spectrum(karr)) * whitenoise
    field = fftp.ifftn(fftp.fftshift(fourier_field)) / volcell

    return field


def compute_isotropic_power_spectrum(field, boxside, binscaling='linear',
                                     kmax=None, nbins=10, retmodes=False):
    """Compute the isotropic power spectrum of a field on a regular grid.

    Parameters
    ----------
    field : float, array_like
        Random field in configuration space.
    boxside : float
        Side length of the Cartesian box (in Mpc/h).
    binscaling : {'linear', 'log'}, optional
        Binning in 'linear' or 'log' scale (default is ``'log'``).
    kmax : float or None, optional
        Maximum wave number (default is `None`).  If `None`, this is set to
        largest wave number the field supports.
    nbins : int or None, optional
        Number of bins each corresponding to a wave number.
    retmodes : bool, optional
        If `True`, also return the number of modes corresponding to each wave
        number of the power spectrum.

    Returns
    -------
    wavenumbers : float, array_like
        Bin-centre wave numbers.
    powers : float, array_like
        Isotropic power spectrum.
    nmodes : int, optional
        Number of modes corresponding to each wave number.

    """
    if len(set(field.shape)) > 1:
        raise ValueError("`field` does not fit on a cubic regular grid. ")

    ncells = max(np.array(field).shape)
    volcell = (boxside/ncells)**3

    _, karr = gen_regular_grid(2*np.pi/boxside, ncells, retnorm=True)
    powerarr = volcell * np.abs(fftp.fftshift(fftp.fftn(field)))**2

    powers, wavenumbers, nmodes = _radial_binning(
        karr, powerarr, nbins, binscaling, rmin=2*np.pi/boxside, rmax=kmax
        )

    if not retmodes:
        return wavenumbers, powers
    return wavenumbers, powers, nmodes


def _radial_binning(norm3d, data3d, nbins, binscaling, rmin=None, rmax=None):

    if rmin is None:
        rmin = np.min(norm3d)
    if rmax is None:
        rmax = np.max(norm3d)

    if binscaling == 'linear':
        bins = np.linspace(rmin, rmax, nbins+1)
    elif binscaling == 'log':
        bins = 10 ** np.linspace(np.log10(rmin), np.log10(rmax), nbins+1)

    counts, _ = np.histogram(norm3d.flatten(), bins=bins)
    bintot, _ = np.histogram(
        norm3d.flatten(), bins=bins, weights=data3d.flatten()
        )

    return bintot/counts, bins[:-1] + np.diff(bins)/2, counts


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from nbodykit.lab import cosmology as cosmo

    import sys  #
    sys.path.insert(0, "../")  #

    from harmonia.collections import harmony

    boxside = 5000.
    nmesh = 256

    Plin = cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')

    field = gen_gaussian_random_field(boxside, nmesh, Plin, seed=81)

    k, Pk = compute_isotropic_power_spectrum(field, boxside, kmax=0.10)

    plt.style.use(harmony)
    plt.close('all')
    plt.figure('Gaussian random field')
    plt.loglog(k, Pk, label='recovered power')
    plt.loglog(k, Plin(k), '--', label='model spectrum')
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
