"""
Random fields (:mod:`~harmonia.algorithms.fields`)
===============================================================================

Generate random fields on 3-d regular grids from input power spectrum in a
cubic box.  Perform biasing, threshold clipping, log-normal transformation and
discrete Poisson sampling of fields.

**Generation**

.. autosummary::

    generate_regular_grid
    generate_gaussian_random_field
    generate_lognormal_random_field

**Transformation**

.. autosummary::

    threshold_clip
    lognormal_transform
    poisson_sample
    populate_particles

|

"""
import warnings

import numpy as np
from scipy import fftpack as fftp
from nbodykit.cosmology.correlation import pk_to_xi, xi_to_pk


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
    grid_coords : list [of length 3] of (N, N, N) array_like, optional
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
                                   clip=True, retdisp=False, seed=None):
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
        Desired power spectrum (in cubic Mpc/h).
    bias : float, optional
        Bias of the density contrast field (default is 1.).
    clip : bool, optional
        If `True` (default), the configuratuion-space field is clipped at
        threshold -1.
    retdisp : bool, optional
        If `True` (default is `False`), also return the velocity displacement
        field for each dimension.
    seed : int or None, optional
        Random seed for the field(s) (default is `None`).

    Returns
    -------
    overdensity : (N, N, N) :class:`numpy.ndarray` of float
        Gaussian random field of density contrast in configuration space.
    displacement : list [of length 3] of (N, N, N) array_like, optional
        Gaussian random fields of velocity displacement in configuration space
        for each dimension.  Returned if `retdisp` is `True`.

    """
    vol, ncell = boxside**3, nmesh**3

    kk, kcoords = generate_regular_grid(2*np.pi/boxside, nmesh, ret='both')
    whitenoise = _gen_circsym_whitenoise(nmesh, seed=seed)

    amplitude = power_spectrum(kk) / vol
    fourier_field = np.sqrt(amplitude) * whitenoise

    overdensity = ncell * np.real(fftp.ifftn(fftp.fftshift(fourier_field)))
    overdensity *= bias

    if clip:
        overdensity = threshold_clip(overdensity)

    if retdisp:
        fourier_disp = [1j * ki/kk**2 * fourier_field for ki in kcoords]

        displacement = [
            ncell * np.real(fftp.ifftn(fftp.fftshift(fourier_disp_i)))
            for fourier_disp_i in fourier_disp
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
    power_spectrum : callable
        Desired power spectrum (in cubic Mpc/h).
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
    displacement : list [of length 3] of (N, N, N) array_like, optional
        Gaussian random fields of velocity displacement in configuration space
        for each dimension.  Returned if `retdisp` is `True`.

    """
    vol, ncell = boxside**3, nmesh**3

    kk, kcoords = generate_regular_grid(2*np.pi/boxside, nmesh, ret='both')
    whitenoise = _gen_circsym_whitenoise(nmesh, seed=seed)

    # Apply biasing at target power spectrum level.
    power_tar = lambda k: bias**2 * power_spectrum(k)

    # Transform target power spectrum.
    amplitude_tar = power_tar(kk) / vol
    xi_tar = ncell * np.real(fftp.ifftn(fftp.fftshift(amplitude_tar)))
    xi_gen = lognormal_transform(xi_tar, 'xi')
    amplitude_gen = fftp.fftshift(fftp.fftn(xi_gen)) / ncell

    fourier_field_gen = np.sqrt(amplitude_gen) * whitenoise

    field_gen = ncell * np.real(fftp.ifftn(fftp.fftshift(fourier_field_gen)))
    field_tar = lognormal_transform(field_gen, 'delta')

    overdensity = field_tar

    # Fulfill continuity equation and no velocity bias.
    if retdisp:

        fourier_field_tar = fftp.fftshift(fftp.fftn(field_tar))  # / ncell
        fourier_disp = [1j * ki/kk**2 * fourier_field_tar for ki in kcoords]

        displacement = [
            np.real(fftp.ifftn(fftp.fftshift(fourier_disp_i)))  # * ncell
            for fourier_disp_i in fourier_disp
            ]

        return overdensity, displacement

    return overdensity


def threshold_clip(density_contrast, threshold=-1.):
    """Apply threshold clipping to density contrast field in configuration
    space.

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

    """
    veto_mask = density_contrast < -1.
    density_contrast[veto_mask] = -1.

    veto_ratio = np.sum(veto_mask) / np.size(veto_mask)
    if veto_ratio > 5e-3:
        warnings.warn(
            "{:.2g}% of field values are clipped. ".format(100*veto_ratio),
            RuntimeWarning
            )

    return density_contrast


def lognormal_transform(obj, objtype):
    r"""Perform log-normal transform of a statistically homogeneous and
    isotropic field or its 2-point functions in either configuration or Fourier
    space.

    Parameters
    ----------
    obj : :class:`numpy.ndarray` of float or callable
        The object to be transformed.
    objtype : {'delta', 'xi', 'pk'}
        Type of the object to be transformed: field (``'delta'``), correlation
        function (``'xi'``) or power spectrum (``'pk'``).

    Returns
    -------
    :class:`numpy.ndarray` of float or callable
        Log-normal transformed object.

    Raises
    ------
    TypeError
        If `objtype` is ``'pk'`` (power spectrum) but `obj` is not callable.
    ValueError
        If `objtype` is not one of the following: ``'delta'``, ``'xi'`` or
        ``'pk'``.

    """
    NPOINT = 1024
    LOG_KMIN, LOG_KMAX = -5, 1
    LOG_RMIN, LOG_RMAX = -1, 5

    k_samples = np.logspace(LOG_KMIN, LOG_KMAX, NPOINT)
    r_samples = np.logspace(LOG_RMIN, LOG_RMAX, NPOINT)

    if objtype.lower().startswith('d'):
        field_var = np.sum(obj**2) / np.size(obj)
        transformed_field = -1 + np.exp(obj - field_var/2)
        return transformed_field

    if objtype.lower().startswith('x'):
        if hasattr(obj, '__call__'):
            return lambda r: np.log(1 + obj(r))
        return np.log(1 + obj)

    if objtype.lower().startswith('p'):
        if not hasattr(obj, '__call__'):
            raise TypeError("Input 2-point function is not of callable type. ")

        Pk_tar_samples = obj(k_samples)
        xi_tar = pk_to_xi(k_samples, Pk_tar_samples)

        xi_gen = lambda r: np.log(1 + xi_tar(r))

        xi_gen_samples = xi_gen(r_samples)
        Pk_gen = xi_to_pk(r_samples, xi_gen_samples)

        return Pk_gen

    raise ValueError(
        "`objtype` must be correlation function 'xi', power spectrum 'pk', "
        "or random field 'delta'. "
        )


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


def populate_particles(sampled_field, mean_density, boxside, voff_fields=None,
                       seed=None):
    """Uniformly place particle at positions within grid cells from a
    discretely sampled field.

    Parameters
    ----------
    sampled_field : (N, N, N) :class:`numpy.ndarray` of float
        Discretely sampled density contrast field.
    mean_density : float
        Overall mean number density of particles.
    boxside : float
        Box size per dimension.
    voff_fields : list [of length 3] of (N, N, N) array_like or None, optional
        Particle velocity offset field (default is `None`).
    seed : int or None, optional
        Particle placement random seed (default is `None`).

    Returns
    -------
    position : (N, 3) :class:`numpy.ndarray` of float
        Position of particles generated from the sampled field.
    displacements : (N, 3) :class:`numpy.ndarray` of float
        Displacements of particles from their `position`.

    """
    if len(set(sampled_field.shape)) > 1:
        raise ValueError("Input `field` is not a regular grid. ")
    nmesh = max(sampled_field.shape)
    cellsize = boxside / nmesh
    volcell = cellsize**3

    grid_coords = generate_regular_grid(cellsize, nmesh, ret='coords')
    cellpos = np.transpose([np.ravel(coords) for coords in grid_coords])

    number_field = np.around(
        (1 + sampled_field) * mean_density * volcell
        ).astype(int)
    position = np.repeat(cellpos, np.ravel(number_field), axis=0)

    np.random.seed(seed=seed)
    position += cellsize * np.random.uniform(
        low=-0.5, high=0.5, size=position.shape
        )

    if voff_fields is not None:
        celldsp = np.transpose([np.ravel(psi_i) for psi_i in voff_fields])
        displacements = np.repeat(celldsp, np.ravel(number_field), axis=0)

        return position, displacements

    return position


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
        Binning range.  If `None` (default), the values are respectively set to
        zero and the largest norm value that the grid supports.

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


def _cal_isotropic_power_spectrum(field, boxside, kmax=None, nbins=10,
                                  binscaling='linear'):
    """Calculate the isotropic power spectrum of a random field in
    configuration space.

    Parameters
    ----------
    field : (N, N, N) :class:`numpy.ndarray` of float
        Random field.
    boxside : float
        Box size per dimension (in Mpc/h).
    kmax : float or None, optional
        Maximum wave number.  If `None` (default), this is set to largest wave
        number the field supports.
    nbins : int or None, optional
        Number of bins each corresponding to a wave number (default is 10).
    binscaling : {'linear', 'log'}, optional
        Binning in 'linear' (default) or 'log' scale.

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
    vol, ncell = boxside**3, nmesh**3

    knorm = generate_regular_grid(2*np.pi/boxside, nmesh, ret='norm')
    powerarr = vol * np.abs(fftp.fftshift(fftp.fftn(field)))**2 / ncell**2

    powers, wavenumbers, nmodes = _radial_binning(
        knorm, powerarr, nbins, binscaling, rmin=0., rmax=kmax
        )

    return wavenumbers, powers, nmodes


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
