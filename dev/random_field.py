"""
Random field generator (:mod:`~harmonia.algorithms.random_field`)
===============================================================================

Generate random fields on 3-d regular grids from input power spectrum in a
cubic box.  Perform biasing, threshold clipping, log-normal transformation and
discrete Poisson sampling of random fields.

**Generation**

.. autosummary::

    generate_regular_grid
    generate_gaussian_random_field
    generate_lognormal_random_field

**Operation**

.. autosummary::

    threshold_clip
    lognormal_transform
    poisson_sample
    particle_populate

|

"""
import warnings

import numpy as np
from scipy import fftpack as fftp
from nbodykit.cosmology.correlation import (
    pk_to_xi as power_to_corr,
    xi_to_pk as corr_to_power
    )


def _gen_circsym_whitenoise(nmesh, seed=None):
    """Generate white noise samples drawn from the circularly-symmetric complex
    normal distribution on a 3-d regular grid.

    Both the real and imaginary parts follow the standard normal distribution,
    so the complex samples have variance 2.

    Parameters
    ----------
    nmesh : int
        Mesh number per dimension.
    seed : int or None, optional
        Random seed (default is `None`).

    Returns
    -------
    whitenoise : (N, N, N) :class:`numpy.ndarray` of complex
        Circularly-symmetric Gaussian noise with double unit variance.

    """
    size = (2,) + (nmesh,)*3
    samples = np.random.RandomState(seed=seed).normal(size=size)

    whitenoise = samples[0] + 1j*samples[1]

    return whitenoise


def generate_regular_grid(cellsize, nmesh, ret='norm'):
    """Generate 3-d coordinate grids for the given cell size and mesh number
    per dimension.

    Parameters
    ----------
    cellsize : float
        Cell size per dimension.
    nmesh : int
        Mesh number per dimension.
    ret : {'norm', 'coords', 'both'}, optional
        The grid to be returned: 'norm' of the grid coordinates (default), or
        'coords' for the grid, or 'both' in that order.

    Returns
    -------
    grid_norm : (N, N, N) :class:`numpy.ndarray` of float, optional
        Grid coordinate norm array.  Returned if `ret` is ``'norm'`` or
        ``'both'``.
    grid_coords : list [of length 3] of (N, N, N) of float array_like, optional
        Grid coordinate arrays for each dimension.  Returned if `ret` is
        ``'coords'`` or ``'both'``.

    Raises
    ------
    ValueError
        If `ret` is not any of the following: ``'norm'``, ``'coords'`` or
        ``'both'``.

    """
    indices = np.indices((nmesh,)*3)
    origin = np.array([(nmesh-1)/2]*3)

    grid_coords = [
        cellsize * (index - centre)
        for index, centre in zip(indices, origin)
        ]
    grid_norm = np.sqrt(sum([coord**2 for coord in grid_coords]))

    if ret.lower().startswith('n'):
        return grid_norm
    if ret.lower().startswith('c'):
        return grid_coords
    if ret.lower().startswith('b'):
        return grid_norm, grid_coords

    raise ValueError("Invalid `ret` array: unknown coordinate grid. ")


def generate_gaussian_random_field(boxside, nmesh, power_spectrum, bias=1.,
                                   retdisp=False, seed=None):
    r"""Generate a Gaussian random field corresponding to the density contrast
    :math:`\delta(\mathbf{r})` in configuration space with matching input power
    spectrum, and optionally a second derived random vector field corresponding
    to the velocity displacement :math:`\boldmath{\Psi}(\mathbf{r})`.

    In Fourier space, the displacement field is related to the density contrast
    field by

    .. math::

        \boldmath{\Psi}(\mathbf{k}) = \mathrm{i} \frac{\mathbf{k}}{k^2}
        \delta(\mathbf{k}) \,,

    and to the velocity field by

    .. math::

        \mathbf{v}(\mathbf{k}) = faH \Psi(\mathbf{k}) \,.

    Parameters
    ----------
    boxside : float
        Box size per dimension (in Mpc/h).
    nmesh : int
        Mesh number per dimension.
    power_spectrum : callable
        Input power spectrum (in cubic Mpc/h).
    bias : float, optional
        Bias of the density contrast field (default is 1.).
    retdisp : bool, optional
        If `True` (default is `False`), also return the velocity displacement
        field for each dimension.
    seed : int or None, optional
        Random seed for the field(s) (default is `None`).

    Returns
    -------
    overdensity : (N, N, N) :class:`numpy.ndarray` of float
        Gaussian random field of density contrast in configuration space.
    displacement : list of [length 3] of (N, N, N) float array_like, optional
        Gaussian random fields of velocity displacement in configuration space
        for each dimension.  Returned if `retdisp` is `True`.

    """
    vol, ncell = boxside**3, nmesh**3
    volcell = vol / ncell

    knorm, kcoor = generate_regular_grid(2*np.pi/boxside, nmesh, ret='both')
    whitenoise = _gen_circsym_whitenoise(nmesh, seed=seed)

    fourier_field = np.sqrt(power_spectrum(knorm)*vol) * whitenoise

    overdensity = np.real(fftp.ifftn(fftp.fftshift(fourier_field))) / volcell
    overdensity *= bias

    if retdisp:
        fourier_disp = [1j * ki/knorm**2 * fourier_field for ki in kcoor]

        displacement = [
            np.real(fftp.ifftn(fftp.fftshift(fourier_disp))) / volcell
            ]

        return overdensity, displacement

    return overdensity


def generate_lognormal_random_field(boxside, nmesh, power_spectrum, bias=1.,
                                    retdisp=False, seed=None):
    """Generate a log-normal random field corresponding to the density contrast
    :math:`\delta(\mathbf{r})` in configuration space with matching input power
    spectrum, and optionally a second derived random vector field corresponding
    to the velocity displacement :math:`\boldmath{\Psi}(\mathbf{r})`.

    See :func:`generate_gaussian_random_field` for their relations.

    Parameters
    ----------
    boxside : float
        Box size per dimension (in Mpc/h).
    nmesh : int
        Mesh number per dimension.
    power_spectrum : :class:`nbodykit.cosmology.power.linear.LinearPower`
        ``nbodykit`` linear power spectrum (in cubic Mpc/h).
    bias : float, optional
        Bias of the density contrast field (default is 1.).
    retdisp : bool, optional
        If `True` (default is `False`), also return the velocity displacement
        fields for each dimension.
    seed : int or None, optional
        Random seed for the field(s) (default is `None`).

    Returns
    -------
    overdensity : (N, N, N) :class:`numpy.ndarray` of float
        Gaussian random field of density contrast in configuration space.
    displacement : list of [length 3] of (N, N, N) float array_like, optional
        Gaussian random fields of velocity displacement in configuration space
        for each dimension.  Returned if `retdisp` is `True`.

    """
    KMIN, KMAX, NPOINT = 1e-5, 10, 1024

    k_samples = np.linspace(KMIN, KMAX, NPOINT)
    r_samples = 2*np.pi / k_samples

    Pk_target = power_spectrum(k_samples)
    xi_target = power_to_corr(k_samples, Pk_target, extrap=True)(r_samples)

    xi_gen = lognormal_transform(xi_target, '2pt')
    power_gen = corr_to_power(r_samples, xi_gen, extrap=True)

    field_gen = generate_gaussian_random_field(
        boxside, nmesh, power_gen, bias=1., retdisp=retdisp, seed=seed
        )
    field_target = lognormal_transform(field_gen, '1pt')

    overdensity = field_target

    return overdensity, displacement


def lognormal_transform(obj, objtype, bias=1.):
    r"""Perform log-normal transform of a statistically homogeneous and
    isotropic field or its 2-point functions in either configuration or Fourier
    space.

    Parameters
    ----------
    obj : :class:`numpy.ndarray` of float or callable
        The object to be transformed.
    objtype : {'delta', 'xi', 'pk'}
        Transform either the field (``'delta'``), the correlation function
        (``'xi'``) or the power spectrum (``'pk'``).
    bias : float, optional
        Bias to be applied to the field (default is 1.).

    Returns
    -------
    :class:`numpy.ndarray` of float or callable
        Log-normal transformed object.

    Raises
    ------
    TypeError
        If `objtype` is ``'xi'`` or ``'pk'`` (i.e. 2-point functions) but
        `obj` is not callable.
    ValueError
        If `objtype` is one ``'delta'``, ``'xi'`` or ``'pk'``.

    """
    KMIN, KMAX, NPOINT = 1e-5, 10, 1024

    k_samples = np.linspace(KMIN, KMAX, NPOINT)
    r_samples = 2*np.pi / k_samples

    if objtype.lower().startswith('x') or objtype.lower().startswith('p'):
        if not hasattr(obj, '__call__'):
            raise TypeError("Input 2-point function is not of type callable. ")

    # FIXME: No biasing implemented.
    if objtype.lower().startswith('d'):
        return

    if objtype.lower().startswith('x'):
        def xi_gen(r):
            return np.log(1 + bias**2 * obj(r))
        return xi_gen

    if objtype.lower().startswith('p'):
        Pk_samples = obj(k_samples)

        xi_target = power_to_corr(k_samples, Pk_samples, extrap=True)
        xi_gen = lambda r: np.log(1 + xi_target(r))

        xi_samples = xi_gen(r_samples)

        Pk_gen = corr_to_power(r_samples, xi_samples, extrap=True)

        return Pk_gen

    raise ValueError(
        "`objtype` must be correlation function 'xi', power spectrum 'pk', "
        "or random field 'delta'. "
        )


def threshold_clip(density_contrast, threshold=-1.):
    """Apply threshold clipping to density contrast field in configuration
    space.

    The number of clipped field values should not exceed 1% of the total.

    Parameters
    ----------
    density_contrast :  (N, N, N) :class:`numpy.ndarray` of float
        Density contrast field.
    threshold : float, optional
        Threshold below which the field values is clipped (default is -1.).

    Returns
    -------
    density_contrast : (N, N, N) :class:`numpy.ndarray` of float
        Clipped density contrast field.

    Raises
    ------
    RuntimeError
        If more than 1% of field values are clipped.

    """
    veto_mask = density_contrast < -1.
    density_contrast[veto_mask] = -1.

    veto_ratio = np.sum(veto_mask) / np.size(veto_mask)
    if veto_ratio > 0.:
        warnings.warn(
            "{:g}% of field values are clipped. ".format(100*veto_ratio),
            RuntimeWarning
            )
#    if veto_ratio > 0.01:
#        raise RuntimeError("Too many field values (over 1%) are clipped. ")

    return density_contrast


def poisson_sample(density_contrast, mean_density, boxside, seed=None):
    """Poisson sample a density contrast field.

    Parameters
    ----------
    density_contrast : (N, N, N) :class:`numpy.ndarray` of float
        Density contrast field being sampled.
    mean_density : float
        Overall mean number density of particles.
    boxside : float
        Box size per dimension.
    seed : int or None, optional
        Sampling random seed (default is `None`).

    Returns
    -------
    sampled_field : :class:`numpy.ndarray` of float
        Poisson sampled density contrast field.

    """
    if len(set(density_contrast.shape)) > 1:
        raise ValueError("`density_contrast` field is not a regular grid. ")
    nmesh = max(density_contrast.shape)
    mean_num_per_cell = mean_density * (boxside/nmesh)**3

    np.random.seed(seed=seed)
    number_field = np.random.poisson(
        lam=(1+density_contrast)*mean_num_per_cell
        )
    sampled_field = number_field / mean_num_per_cell - 1

    return sampled_field


def particle_populate(sampled_field, mean_density, boxsize, seed=None):
    """Uniformly place particle at positions within grid cells from a
    discretely sampled field.

    Parameters
    ----------
    sampled_field : (N, N, N) :class:`numpy.ndarray` of float
        Discretely sampled density contrast field.
    mean_density : float
        Overall mean number density of particles.
    boxsize : float
        Box size per dimension.
    seed : int or None, optional
        Particle placement random seed (default is `None`).

    Returns
    -------
    position : (N, 3) :class:`numpy.ndarray` of float
        Position of particles generated from the sampled field.

    """
    if len(set(sampled_field.shape)) > 1:
        raise ValueError("Input `field` is not a regular grid. ")
    nmesh = max(sampled_field.shape)
    cellsize = boxsize / nmesh
    volcell = cellsize**3

    grid_coords = generate_regular_grid(cellsize, nmesh, ret='coords')
    cellpos = np.transpose([np.ravel(coords) for coords in grid_coords])

    number_field = np.around((1 + sampled_field) * mean_density * volcell)
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
    field : (N, N, N) array_like float
        Random field.
    boxside : float
        Box size per dimension (in Mpc/h).
    kmax : float or None, optional
        Maximum wave number (default is `None`).  If `None`, this is set to
        largest wave number the field supports.
    nbins : int or None, optional
        Number of bins each corresponding to a wave number.
    binscaling : {'linear', 'log'}, optional
        Binning in 'linear' or 'log' (default) scale.

    Returns
    -------
    wavenumbers : float, array_like
        Bin-averaged wave numbers.
    powers : float, array_like
        Radially averaged power spectrum at bin wave numbers.
    nmodes : int, array_like
        Number of modes in bins corresponding to each wave number (double
        counting for wave vector parity).

    """
    nmesh = max(np.array(field).shape)
    vol = boxside**3
    volcell = vol / nmesh**3

    knorm = generate_regular_grid(2*np.pi/boxside, nmesh, ret='norm')
    powerarr = volcell**2 / vol * np.abs(fftp.fftshift(fftp.fftn(field)))**2

    powers, wavenumbers, nmodes = _radial_binning(
        knorm, powerarr, nbins, binscaling, rmin=2*np.pi/boxside, rmax=kmax
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
        bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)

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
    field_seed, sampling_seed, drift_seed = \
        np.random.RandomState().randint(low=0, high=4294967295, size=3)

    # -- Generation -----------------------------------------------------------

    # Planck15 cosmology
    Plin = cosmo.LinearPower(cosmology, redshift=Z, transfer='CLASS')

    gaussian_field = generate_gaussian_random_field(
        BOXSIZE, NMESH, Plin, bias=BIAS, seed=field_seed
        )

    gaussian_field = threshold_clip(gaussian_field)

    gaussian_realisation = poisson_sample(
        gaussian_field, NBAR, BOXSIZE, seed=sampling_seed
        )

    # -- Evaluation -----------------------------------------------------------

    k, Pk_gauss, nmodes = _cal_isotropic_power_spectrum(
        gaussian_realisation, BOXSIZE, kmax=KMAX
        )

    Pk = BIAS**2 * Plin(k) + 1/NBAR

    # -- Visualisation --------------------------------------------------------

    plt.style.use(harmony)
    plt.close('all')
    plt.figure('Unsampled biased random field')

    Pk_gauss *= 2.556073158732712
    plt.errorbar(
        k, Pk_gauss, yerr=np.sqrt(2/nmodes)*Pk_gauss,
        label='Gaussian realisation'
        )
    plt.loglog(k, Pk, '--', label='power spectrum with shot noise')

    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
