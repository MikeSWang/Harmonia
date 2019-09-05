"""
Random field generator (:mod:`~harmonia.algorithms.random_field`)
===============================================================================

Generate random fields on 3-d regular grids from input power spectrum.  Perform
biasing, threshold clipping, log-normal transformation and discrete Poisson
sampling of random fields.

**Generation**

.. autosummary::

    generate_regular_grid
    generate_gaussian_random_fields
    generate_lognormal_random_fields

**Operation**

.. autosummary::

    perform_biased_clipping
    perform_lognormal_transformation
    perform_poisson_sampling
    perform_particle_population

|

"""
import warnings

import numpy as np
from scipy import fftpack as fftp
from nbodykit.cosmology import correlation as corr


def _gen_complex_whitenoise(ncell, seed=None):
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
        Coordinate norm array for the grid.  Returned if `retnorm` is `True`.

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


# TODO: Implement velocity displacement field.
def generate_gaussian_random_fields(boxside, ncell, power_spectrum, seed=None):
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
    contrast : :class:`numpy.ndarray` of float
        Gaussian random contrast field in configuration space.
    displacement : :class:`numpy.ndarray` of float
        Gaussian random velocity displacement field in Fourier space.

    """
    whitenoise = _gen_complex_whitenoise(ncell, seed=seed)

    volcell = (boxside/ncell)**3
    _, karr = generate_regular_grid(2*np.pi/boxside, ncell, retnorm=True)

    fourier_field = np.sqrt(volcell * power_spectrum(karr)) * whitenoise
    contrast = np.sqrt(2) / volcell * np.real(
        fftp.ifftn(fftp.fftshift(fourier_field))
        )
    displacement = None

    return contrast, displacement


def generate_lognormal_random_fields(boxside, ncell, power_spectrum, bias=1.,
                                     seed=None):
    """Generate a log-normal random contrast field :math:`\delta(\mathbf{r})`
    in configuration space matching an input power spectrum, with a second
    random field in Fourier space being the displacement field
    math:`\Psi(\mathbf{k})`.

    Parameters
    ----------
    boxside : float
        Box size per dimension (in Mpc/h).
    ncell : int
        Number of cells per dimension.
    power_spectrum : :class:`nbodykit.cosmology.power.linear.LinearPower`
        Input power spectrum (in cubic Mpc/h).
    bias : float, optional
        Bias of the generated field (default is 1.).
    seed : int or None, optional
        Global random seed for the field (default is `None`).

    Returns
    -------
    contrast : :class:`numpy.ndarray` of float
        Log-normal random contrast field in configuration space.
    displacement : :class:`numpy.ndarray` of float
        Log-normal random velocity displacement field in Fourier space.

    """
    correlation = corr.CorrelationFunction(power_spectrum)

    working_r = 10**np.linspace(0, 5, 10**4+1)
    working_correlation = perform_lognormal_transformation(
        correlation, 'xi'
        )

    working_power = corr.xi_to_pk(working_r, working_correlation(working_r))

    working_contrast, displacement = generate_gaussian_random_fields(
        boxside, ncell, working_power, seed=seed
        )

    contrast = perform_lognormal_transformation(
        working_contrast, 'delta', bias=bias
        )

    return contrast, displacement


def perform_lognormal_transformation(obj, objtype, bias=1.):
    """Perform log-normal transform from target to auxiliary fields,
    correlation function or power spectrum.

    Parameters
    ----------
    obj : :class:`numpy.ndarray` of float or callable
        Object to be transformed.
    objtype : {'xi', 'pk', 'delta'}
        Object type can be correlation function 'xi', power spectrum 'pk' or
        random field 'delta'.
    bias : float, optional
        Bias to be applied to the field (default is 1.).

    Returns
    -------
    :class:`numpy.ndarray` of float or callable
        Log-normal transformed object.

    """

    if objtype.lower().startswith('d'):
        field = obj

        biased_field = bias * field
        field_var = np.var(biased_field)

        transformed_field = np.exp(-field_var + biased_field) - 1

        return transformed_field

    if objtype.lower().startswith('x') or objtype.lower().startswith('c'):
        if not hasattr(obj, '__call__'):
            raise TypeError("Input correlation function is not callable. ")
        xi = obj

        def transformed_xi(r):
            return np.log(1 + xi(r))

        return transformed_xi

    raise ValueError(
        "`objtype` must be correlation function 'xi', power spectrum 'pk', "
        "or random field 'delta'. "
        )


def perform_biased_clipping(contrast_field, bias=1.):
    """Apply bias and clipping to contrast random fields in configuration
    space.

    The clipping threshold is -1 and the number of clipped field values should
    not exceed 1% of the total.

    Parameters
    ----------
    contrast_field : :class:`numpy.ndarray` of float
        Contrast random field.
    bias : float, optional
        Bias to be applied to the field (default is 1.).

    Returns
    -------
    biased_field : :class:`numpy.ndarray` of float
        Clipped biased random field.

    Raises
    ------
    RuntimeError
        If more than 1% of field values are clipped to -1.

    """
    biased_field = bias * contrast_field

    veto_mask = contrast_field <= -1.
    veto_ratio = np.sum(veto_mask) / np.size(veto_mask)

    if veto_ratio > 0.:
        warnings.warn(
            "{:g}% of field values are clipped. ".format(100*veto_ratio),
            RuntimeWarning
            )
    if veto_ratio > 0.01:
        raise RuntimeError("Too many field values (over 1%) are clipped. ")

    biased_field[veto_mask] = -1.

    return biased_field


def perform_poisson_sampling(contrast_field, density, boxsize, seed=None):
    """Poisson sample a contrast field to particles randomly placed within
    grid cells.

    Parameters
    ----------
    contrast_field : :class:`numpy.ndarray` of float
        Contrast field being sampled.
    density : float
        Mean number density for grid cells overall.
    boxsize : float
        Box size as a scalar.
    seed : int or None, optional
        Sampling random seed (default is `None`).

    Returns
    -------
    sampled_field : :class:`numpy.ndarray` of float
        Poisson sampled field.

    """
    if len(set(contrast_field.shape)) > 1:
        raise ValueError("Input `field` does not fit on a cubic grid. ")
    ncell = max(contrast_field.shape)
    volcell = (boxsize/ncell)**3

    np.random.seed(seed=seed)
    number_field = np.random.poisson(lam=(1+contrast_field)*density) * volcell
    sampled_field = number_field / (density * volcell) - 1

    return sampled_field


def perform_particle_population(sampled_field, density, boxsize, seed=None):
    """Uniformly place particle at positions within grid cells from a
    discretely sampled field.

    Parameters
    ----------
    sampled_field : :class:`numpy.ndarray` of float
        Sampled contrast field.
    density : float
        Overall mean number density for the sampled field.
    boxsize : float
        Box size as a scalar.
    seed : int or None, optional
        Uniform placement random seed (default is `None`).

    Returns
    -------
    position : :class:`numpy.ndarray` of float
        Position of particles generated from the sampled field.

    """
    if len(set(sampled_field.shape)) > 1:
        raise ValueError("Input `field` does not fit on a cubic grid. ")
    ncell = max(sampled_field.shape)
    cellsize = boxsize / ncell
    volcell = cellsize**3

    coords_list, _ = generate_regular_grid(cellsize, ncell, retnorm=True)
    cellpos = np.transpose([np.ravel(coords) for coords in coords_list])

    number_field = np.around((1 + sampled_field)*density*volcell)
    position = np.repeat(cellpos, np.ravel(number_field), axis=0)

    np.random.seed(seed=seed)
    position += cellsize * np.random.uniform(
        low=-0.5, high=0.5, size=position.shape
        )

    return position


def _cal_isotropic_power_spectrum(field, boxside, kmax=None, nbins=10,
                                      binscaling='linear'):
    """Calculate the isotropic power spectrum of a random field in
    configuration space.

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

    # -- Parameters -----------------------------------------------------------

    NBAR = 1e-3
    BIAS = 2.
    Z = 0.
    KMAX = 0.1
    BOXSIZE = 1000.
    NMESH = 256

    cosmology = cosmo.Planck15
    seed = np.random.randint(0, 4294967295)

    # -- Generation -----------------------------------------------------------

    # Planck15 cosmology
    Plin = cosmo.LinearPower(cosmology, redshift=Z, transfer='CLASS')

    gaussian_field, _ = generate_gaussian_random_fields(
        BOXSIZE, NMESH, Plin, seed=seed
        )
    gaussian_field = perform_biased_clipping(gaussian_field, bias=BIAS)

    lognormal_field, _ = generate_lognormal_random_fields(
        BOXSIZE, NMESH, Plin, bias=BIAS, seed=seed)

    gaussian_realisation = perform_poisson_sampling(
        gaussian_field, NBAR, BOXSIZE
        )
    lognormal_realisation = perform_poisson_sampling(
        lognormal_field, NBAR, BOXSIZE
        )

    # -- Evaluation -----------------------------------------------------------

    k, Pk_gauss, nmodes = _cal_isotropic_power_spectrum(
        gaussian_field, BOXSIZE, kmax=KMAX
        )
    k, Pk_lognorm, nmodes = _cal_isotropic_power_spectrum(
        lognormal_field, BOXSIZE, kmax=KMAX
        )

    Pk = BIAS**2 * Plin(k)

    # -- Visualisation --------------------------------------------------------

    plt.style.use(harmony)
    plt.close('all')
    plt.figure('Unsampled biased random fields')

    plt.errorbar(
        k, Pk_gauss, yerr=np.sqrt(2/nmodes)*Pk_gauss, label='Gaussian'
        )
    plt.errorbar(
        k, Pk_lognorm+5000, yerr=np.sqrt(2/nmodes)*Pk_lognorm, label='log-normal'
        )
    plt.loglog(k, Pk, '--', label='power spectrum with shot noise')

    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
