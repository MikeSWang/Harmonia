"""
Random field generator (:mod:`~harmonia.algorithms.randomfield`)
===============================================================================

Generate random fields from input power spectrum and compute power spectrum of
an input random field.  Perform log-normal transformations and Poisson discrete
sampling of random fields.  Random fields are treated as 3-d arrays over
regular grid cells in a cubic box.

**Generation**

.. autosummary::

    generate_regular_grid
    generate_complex_whitenoise
    generate_gaussian_randomfields

**Operation**

.. autosummary::

    perform_lognormal_transformation
    perform_poisson_sampling

|

"""
import numpy as np
from scipy import fftpack as fftp


def generate_regular_grid(cellsize, ncell, retnorm=False):
    """Generate 3-d coordinate grids for given cell number and cell size per
    dimension.

    Parameters
    ----------
    cellsize : float
        Cell size per dimension.
    ncell : int
        Number of cells per dimension.
    retnorm : bool, optional
        If `True` (default is `False`), also return the coordinate vector norm
        array for the grid.

    Returns
    -------
    coords_list : list [of length 3] of 3-d :class:`numpy.ndarray` of float
        Coordinate arrays for each dimension of the grid.
    normgrid : 3-d :class:`numpy.ndarray` of float, optional
        Coordinate vector norm array for the grid.  Returned if `retnorm` is
        `True`.

    """
    indxarr = np.indices((ncell,)*3)
    centres = np.array([(ncell-1)/2]*3)

    coords_list = [
        cellsize * (indx - centre)
        for indx, centre in zip(indxarr, centres)
        ]

    if not retnorm:
        return coords_list

    normgrid = np.sqrt(sum([coords**2 for coords in coords_list]))

    return coords_list, normgrid


def generate_complex_whitenoise(ncell, seed=None):
    """Generate white noise drawn from the standard complex normal
    distribution.

    Parameters
    ----------
    ncell : int
        Number of cells per dimension.
    seed : int or None, optional
        Global random seed (default is `None`).

    Returns
    -------
    whitenoise : :class:`numpy.ndarray` of complex
        Complex white noise.

    """
    if seed is not None:
        np.random.seed(seed)

    whitenoise_real = np.random.normal(size=(ncell,)*3)
    whitenoise_imag = np.random.normal(size=(ncell,)*3)

    whitenoise = (whitenoise_real + 1j*whitenoise_imag) / np.sqrt(2)

    return whitenoise


# TODO: Implement velocity displacement field.
def generate_gaussian_randomfields(boxside, ncell, power_spectrum, seed=None):
    r"""Generate a Gaussian random contrast field :math:`\delta(\mathbf{r})` in
    configuration space matching an input power spectrum, with a second
    Gaussian random field in Fourier space being the displacement field

    .. math::

        \boldmath{\Psi}(\mathbf{k}) = \mathrm{i} \frac{\mathbf{k}}{k^2}
        \delta(\mathbf{k})

    related to the linear-order velocity field

    .. math::

        \mathbf{v}(\mathbf{k}) = fHa \Psi(\mathbf{k}) \,.

    Parameters
    ----------
    boxside : float
        Box size per dimension (in Mpc/h).
    ncell : int
        Number of cells per dimension.
    power_spectrum : callable
        Input power spectrum (in cubic Mpc/h).
    seed : int or None, optional
        Global random seed for the field (default is `None`).

    Returns
    -------
    contrast_field : :class:`numpy.ndarray` of float
        Gaussian random field.
    displacement_field : :class:`numpy.ndarray` of float
        Gaussian random field.

    """
    whitenoise = generate_complex_whitenoise(ncell, seed=seed)

    volcell = (boxside/ncell)**3
    _, karr = generate_regular_grid(2*np.pi/boxside, ncell, retnorm=True)

    fourier_field = np.sqrt(volcell * power_spectrum(karr)) * whitenoise
    contrast_field = np.sqrt(2) * np.real(
        fftp.ifftn(fftp.fftshift(fourier_field))
        ) / volcell
    displacement_field = None

    return contrast_field, displacement_field


def perform_lognormal_transformation(gaussian_random_field, bias=1.):
    """Log-normal transform a Gaussian random field in configuration space.

    Parameters
    ----------
    gaussian_random_field : :class:`numpy.ndarray` of float
        Gaussian random field.
    bias : float, optional
        Bias applied to the field (default is 1.).

    Returns
    -------
    transformed_field : :class:`numpy.ndarray` of float
        Log-normal random field.

    """
    field = bias * gaussian_random_field
    field_var = np.var(field)

    transformed_field = np.exp(-field_var + field) - 1

    return transformed_field


def perform_poisson_sampling(contrast_field, poisson_mean, boxsize, seed=None):
    """Poisson sample a contrast field to particles randomly placed within
    grid cells.

    Parameters
    ----------
    contrast_field : :class:`numpy.ndarray` of float
        Contrast field being sampled.
    poisson_mean : float
        Poisson mean for each grid cell.
    boxsize : (list of [length 3]) of float
        Box size as a scalar or a triple of scalars.
    seed : int or None, optional
        Sampling random seed (default is `None`).

    Returns
    -------
    position : :class:`numpy.ndarray` of float
        Particle positions sampled from the field.

    """
    if len(set(contrast_field.shape)) > 1:
        raise ValueError("Input `field` does not fit on a cubic grid. ")
    ncell = max(contrast_field.shape)
    cellsize = boxsize / ncell
    volcell = cellsize**3

    # Set random seeds.
    poisson_seed, uniform_seed = np.random.RandomState(seed).randint(
        0, 0xfffffff, size=2
        )

    # Poisson sample particle number in each grid cell.
    nparticles = np.around(np.random.poisson(
        lam=poisson_mean * (1+contrast_field) * volcell
        ))

    # Random drift to particles in each grid cell.
    coords_list, _ = generate_regular_grid(cellsize, ncell, retnorm=True)
    cellpos = np.transpose([np.ravel(coords) for coords in coords_list])
    position = np.repeat(cellpos, np.ravel(nparticles), axis=0)
    position += cellsize * np.random.uniform(
        low=-0.5, high=0.5, size=position.shape
        )

    return position


def _compute_isotropic_power_spectrum(field, boxside, kmax=None, nbins=10,
                                      binscaling='linear'):
    """Compute the isotropic power spectrum of a random field in configuration
    space.

    Parameters
    ----------
    field : float, array_like
        Random field.
    boxside : float
        Box size per dimension (in Mpc/h).
    kmax : float or None, optional
        Maximum wave number (default is `None`).  If `None`, this is set to
        largest wave number the field supports.
    nbins : int or None, optional
        Number of bins each corresponding to a wave number.
    binscaling : {'linear', 'log'}, optional
        Binning in 'linear' or 'log' scale (default is ``'log'``).

    Returns
    -------
    wavenumbers : float, array_like
        Bin-averaged wave numbers.
    powers : float, array_like
        Isotropic power spectrum.
    nmodes : int, array_like
        Number of modes corresponding to each wave number.

    """
    ncell = max(np.array(field).shape)
    volcell = (boxside/ncell)**3

    _, karr = generate_regular_grid(2*np.pi/boxside, ncell, retnorm=True)
    powerarr = volcell * np.abs(fftp.fftshift(fftp.fftn(field)))**2

    powers, wavenumbers, nmodes = _radial_binning(
        karr, powerarr, nbins, binscaling, rmin=2*np.pi/boxside, rmax=kmax
        )

    return wavenumbers, powers, nmodes


def _radial_binning(norm3d, data3d, nbins, binscaling, rmin=None, rmax=None):
    """Radial binning by coordinate vector norm for 3-d data over a regular
    grid.

    Parameters
    ----------
    norm3d : float, array_like
        Coordinate vector norm grid.
    data3d : float, array_like
        Data array over the same grid.
    nbins : int
        Number of bins in coordinate norm.
    binscaling : {'linear', 'log'}
        Binning in 'linear' or 'log' scale.
    rmin, rmax : float or None, optional
        Binning range.  If `None`, the values are set to the extreme norm
        values that the grid supports.

    Returns
    -------
    bindat : float, array_like
        Bin-averaged data values.
    bincoord : float, array_like
        Bin-averaged norm values.
    counts : int, array_like, optional
        Number of data points in each bin.

    """
    if rmin is None:
        rmin = np.min(norm3d)
    if rmax is None:
        rmax = np.max(norm3d)

    if binscaling == 'linear':
        bins = np.linspace(rmin, rmax, nbins+1)
    elif binscaling == 'log':
        bins = 10 ** np.linspace(np.log10(rmin), np.log10(rmax), nbins+1)

    counts, _ = np.histogram(norm3d.flatten(), bins=bins)
    bincoord, _ = np.histogram(
        norm3d.flatten(), bins=bins, weights=norm3d.flatten()
        )  # aggregate value
    bindat, _ = np.histogram(
        norm3d.flatten(), bins=bins, weights=data3d.flatten()
        )  # aggregate value

    bincoord /= counts
    bindat /= counts

    return bindat, bincoord, counts


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from nbodykit.lab import cosmology as cosmo

    import sys  #
    sys.path.insert(0, "../")  #
    from harmonia.collections import harmony

    # Parameters
    Z = 0.
    BOXSIZE = 5000.
    NMESH = 256
    KMAX = 0.1
    COSMOLOGY = cosmo.Planck15

    # Generation (Planck15 cosmology)
    Plin = cosmo.LinearPower(COSMOLOGY, redshift=Z, transfer='CLASS')

    gaussian_field, _ = generate_gaussian_randomfields(BOXSIZE, NMESH, Plin)

    lognormal_field = perform_lognormal_transformation(gaussian_field)

    # Evaluation
    k, Pk_gauss, nmodes = _compute_isotropic_power_spectrum(
        gaussian_field, BOXSIZE, kmax=KMAX
        )
    k, Pk_lognorm, nmodes = _compute_isotropic_power_spectrum(
        lognormal_field, BOXSIZE, kmax=KMAX
        )

    Pk = Plin(k)

    # Visualisation
    plt.style.use(harmony)
    plt.close('all')
    plt.figure('Random fields')

    plt.errorbar(
        k, Pk_gauss, yerr=np.sqrt(2/nmodes)*Pk_gauss, label='Gaussian'
        )
    plt.errorbar(
        k, Pk_lognorm, yerr=np.sqrt(2/nmodes)*Pk_lognorm, label='log-normal'
        )
    plt.loglog(k, Pk, '--', label='power spectrum')

    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
