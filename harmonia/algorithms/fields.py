"""
Random fields (:mod:`~harmonia.algorithms.fields`)
===========================================================================

Generate random fields on 3-d regular grids from input power spectrum in a
cubic box, and perform biasing, threshold clipping, log-normal
transformation and discrete Poisson sampling of fields.

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
from nbodykit.cosmology.correlation import pk_to_xi, xi_to_pk
from scipy import fftpack as fftp


def generate_regular_grid(cell_size, num_mesh, variable='norm'):
    """Generate 3-d coordinate grids for the given cell size and mesh
    number per dimension.

    Parameters
    ----------
    cell_size : float
        Cell size per dimension.
    num_mesh : int
        Mesh number per dimension.
    variable : {'norm', 'coords', 'both'}, optional
        The grid variable to be returned: 'norm' (default) of the grid
        coordinates, or 'coords' for the grid, or 'both' in that order.

    Returns
    -------
    grid_norm : float :class:`numpy.ndarray`
        Grid coordinate norm array.  Returned if `variable` is ``'norm'``
        or ``'both'``.
    grid_coords : :obj:`list` of float :class:`numpy.ndarray`
        Grid coordinate arrays for each dimension.  Returned if `variable`
        is ``'coords'`` or ``'both'``.

    Raises
    ------
    ValueError
        If `variable` is not any of the following: ``'norm'``, ``'coords'``
        or ``'both'``.

    """
    indices = np.indices((num_mesh,)*3)
    origin = np.array([(num_mesh-1)/2]*3)

    grid_coords = [
        cell_size * (index - centre) for index, centre in zip(indices, origin)
    ]
    grid_norm = np.sqrt(sum([coord**2 for coord in grid_coords]))

    if variable.lower().startswith('n'):
        return grid_norm
    if variable.lower().startswith('c'):
        return grid_coords
    if variable.lower().startswith('b'):
        return grid_norm, grid_coords
    raise ValueError(f"Unknown grid `variable`: {variable}. ")


def generate_gaussian_random_field(boxsize, num_mesh, power_spectrum, bias=1.,
                                   clip=True, return_disp=False, seed=None):
    r"""Generate a Gaussian random field corresponding to the density
    contrast :math:`\delta(\mathbf{r})` in configuration space with
    matching input power spectrum, and optionally a second derived random
    vector field corresponding to the velocity displacement
    :math:`\boldsymbol{\Psi}(\mathbf{r})`.

    In Fourier space, the displacement field is related to the density
    contrast field by

    .. math::

        \boldsymbol{\Psi}(\mathbf{k}) = \mathrm{i} \frac{\mathbf{k}}{k^2}
        \delta(\mathbf{k}) \,,

    and to the velocity field by

    .. math::

        \mathbf{v}(\mathbf{k}) = faH \Psi(\mathbf{k}) \,.

    Parameters
    ----------
    boxsize : float
        Box size per dimension (in Mpc/h).
    num_mesh : int
        Mesh number per dimension.
    power_spectrum : callable
        Desired power spectrum (in cubic Mpc/h).
    bias : float, optional
        Bias of the density contrast field (default is 1.).
    clip : bool, optional
        If `True` (default), the configuratuion-space field is clipped at
        threshold -1.
    return_disp : bool, optional
        If `True` (default is `False`), also return the velocity
        displacement field for each dimension that is not `None`.
    seed : int or None, optional
        Random seed for the field(s) (default is `None`).

    Returns
    -------
    overdensity : float :class:`numpy.ndarray`
        Gaussian random field of density contrast in configuration space.
    displacement : :obj:`list` of float :class:`numpy.ndarray` or None
        Gaussian random fields of velocity displacement in configuration
        space for each dimension.  Returned as `None` unless `return_disp`
        is `True`.

    """
    vol, num_cell = boxsize**3, num_mesh**3
    k_norm, k_vec = generate_regular_grid(
        2*np.pi / boxsize,
        num_mesh,
        variable='both',
    )

    whitenoise = _gen_circsym_whitenoise(num_mesh, seed=seed)
    amplitude = power_spectrum(k_norm) / vol
    fourier_field = bias * np.sqrt(amplitude) * whitenoise

    overdensity = num_cell * np.real(fftp.ifftn(fftp.fftshift(fourier_field)))
    if clip:
        overdensity = threshold_clip(overdensity)

    if return_disp:
        fourier_disp = [1j * ki / k_norm**2 * fourier_field for ki in k_vec]
        displacement = [
            num_cell * np.real(fftp.ifftn(fftp.fftshift(fourier_disp_i)))
            for fourier_disp_i in fourier_disp
        ]
    else:
        displacement = None

    return overdensity, displacement


def generate_lognormal_random_field(boxsize, num_mesh, power_spectrum, bias=1.,
                                    return_disp=False, seed=None):
    r"""Generate a log-normal random field corresponding to the density
    contrast :math:`\delta(\mathbf{r})` in configuration space with
    matching input power spectrum, and optionally a second derived random
    vector field corresponding to the velocity displacement
    :math:`\boldsymbol{\Psi}(\mathbf{r})`.

    The velocity displacement field fullfils the continuity equation and
    no velocity bias is assumed.

    Parameters
    ----------
    boxsize : float
        Box size per dimension (in Mpc/h).
    num_mesh : int
        Mesh number per dimension.
    power_spectrum : callable
        Desired power spectrum (in cubic Mpc/h).
    bias : float, optional
        Bias of the density contrast field (default is 1.).
    return_disp : bool, optional
        If `True` (default is `False`), also return the velocity
        displacement field for each dimension that is not `None`.
    seed : int or None, optional
        Random seed for the field(s) (default is `None`).

    Returns
    -------
    overdensity : float :class:`numpy.ndarray`
        Gaussian random field of density contrast in configuration space.
    displacement : :obj:`list` of float :class:`numpy.ndarray` or None
        Gaussian random fields of velocity displacement in configuration
        space for each dimension.  Returned as `None` unless `return_disp`
        is `True`.

    See Also
    --------
    :func:`generate_gaussian_random_field`

    """
    vol, num_cell = boxsize**3, num_mesh**3
    k_norm, k_vec = generate_regular_grid(
        2*np.pi / boxsize,
        num_mesh,
        variable='both',
    )

    power_spectrum_target = lambda k: bias**2 * power_spectrum(k)
    amplitude_target = power_spectrum_target(k_norm) / vol
    xi_target = num_cell * np.real(fftp.ifftn(fftp.fftshift(amplitude_target)))
    xi_gen = lognormal_transform(xi_target, 'xi')
    amplitude_gen = fftp.fftshift(fftp.fftn(xi_gen)) / num_cell

    whitenoise = _gen_circsym_whitenoise(num_mesh, seed=seed)
    fourier_field_gen = np.sqrt(amplitude_gen) * whitenoise

    field_gen = num_cell \
        * np.real(fftp.ifftn(fftp.fftshift(fourier_field_gen)))
    field_target = lognormal_transform(field_gen, 'delta')

    overdensity = field_target

    if return_disp:
        fourier_field_target = fftp.fftshift(fftp.fftn(field_target))
        fourier_disp = [
            1j * ki / k_norm**2 * fourier_field_target for ki in k_vec
        ]
        displacement = [
            np.real(fftp.ifftn(fftp.fftshift(fourier_disp_i)))
            for fourier_disp_i in fourier_disp
        ]
    else:
        displacement = None

    return overdensity, displacement


def threshold_clip(density_contrast, threshold=-1.):
    """Apply threshold clipping to density contrast field in configuration
    space.

    Parameters
    ----------
    density_contrast : float, array_like
        Density contrast field.
    threshold : float, optional
        Threshold below which the field values is clipped (default is -1.).

    Returns
    -------
    density_contrast : float :class:`numpy.ndarray`
        Clipped density contrast field.

    """
    clip_mask = density_contrast < -1.
    density_contrast[clip_mask] = -1.

    clip_ratio = np.sum(clip_mask) / np.size(clip_mask)
    if clip_ratio > 0.005:
        warnings.warn(
            "{:.2g}% of field values are clipped. ".format(100*clip_ratio),
            RuntimeWarning,
        )

    return density_contrast


def lognormal_transform(obj, obj_type):
    r"""Perform log-normal transform of a statistically homogeneous and
    isotropic field or its 2-point functions in either configuration or
    Fourier space.

    Parameters
    ----------
    obj : array_like or callable
        The object to be transformed.
    obj_type : {'delta', 'xi', 'pk'}
        Type of the object to be transformed: field (``'delta'``),
        correlation function (``'xi'``) or power spectrum (``'pk'``).

    Returns
    -------
    float :class:`numpy.ndarray` or callable
        Log-normal transformed object.

    Raises
    ------
    TypeError
        If `obj_type` is ``'pk'`` (power spectrum) but `obj` is not
        callable.
    ValueError
        If `obj_type` is not one of the following: ``'delta'``, ``'xi'`` or
        ``'pk'``.

    """
    NUM_POINT = 1024
    LOG_KMIN, LOG_KMAX = -5, 1
    LOG_RMIN, LOG_RMAX = -1, 5

    k_samples = np.logspace(LOG_KMIN, LOG_KMAX, NUM_POINT)
    r_samples = np.logspace(LOG_RMIN, LOG_RMAX, NUM_POINT)

    if obj_type.lower().startswith('d'):
        field_var = np.sum(obj**2) / np.size(obj)
        transformed_field = -1 + np.exp(obj - field_var/2)
        return transformed_field

    if obj_type.lower().startswith('x'):
        if callable(obj):
            return lambda r: np.log(1 + obj(r))
        return np.log(1 + obj)

    if obj_type.lower().startswith('p'):
        if not callable(obj):
            raise TypeError("Input 2-point function is not callable. ")

        Pk_target_samples = obj(k_samples)
        xi_target = pk_to_xi(k_samples, Pk_target_samples)

        xi_gen = lambda r: np.log(1 + xi_target(r))

        xi_gen_samples = xi_gen(r_samples)
        Pk_gen = xi_to_pk(r_samples, xi_gen_samples)

        return Pk_gen

    raise ValueError(
        f"Invalid `obj_type`: {obj_type}. This must be "
        "correlation function 'xi', "
        "power spectrum 'pk', "
        "or random field 'delta'. "
    )


def poisson_sample(density_contrast, mean_density, boxsize, seed=None):
    """Poisson sample a density contrast field.

    Parameters
    ----------
    density_contrast : float, array_like
        Density contrast field being sampled.
    mean_density : float
        Overall mean number density of particles.
    boxsize : float
        Box size per dimension.
    seed : int or None, optional
        Sampling random seed (default is `None`).

    Returns
    -------
    sampled_field : float :class:`numpy.ndarray`
        Poisson sampled density contrast field.

    """
    if len(set(density_contrast.shape)) > 1:
        raise ValueError("`density_contrast` field is not a regular grid. ")
    num_mesh = max(density_contrast.shape)
    mean_num_per_cell = mean_density * (boxsize / num_mesh)**3

    np.random.seed(seed=seed)
    number_field = np.random.poisson(
        lam=(1+density_contrast)*mean_num_per_cell
    )
    sampled_field = number_field / mean_num_per_cell - 1

    return sampled_field


def populate_particles(sampled_field, mean_density, boxsize,
                       vel_offset_fields=None, seed=None):
    """Uniformly place particle at positions within grid cells from a
    discretely sampled field.

    Parameters
    ----------
    sampled_field : float, array_like
        Discretely sampled density contrast field.
    mean_density : float
        Overall mean number density of particles.
    boxsize : float
        Box size per dimension.
    vel_offset_fields : :obj:`list` of float, array_like, optional
        Particle velocity offset field (default is `None`).
    seed : int or None, optional
        Particle placement random seed (default is `None`).

    Returns
    -------
    position : float :class:`numpy.ndarray`
        Position of particles generated from the sampled field.
    displacement : float :class:`numpy.ndarray`
        Displacement of particles from their `position`.

    """
    if len(set(sampled_field.shape)) > 1:
        raise ValueError("`field` is not a regular grid. ")
    num_mesh = max(sampled_field.shape)
    cell_size = boxsize / num_mesh
    vol_cell = cell_size**3

    grid_coords = generate_regular_grid(cell_size, num_mesh, variable='coords')
    cell_position = np.transpose([np.ravel(coords) for coords in grid_coords])

    number_field = np.around(
        (1 + sampled_field) * mean_density * vol_cell
    ).astype(int)
    position = np.repeat(cell_position, np.ravel(number_field), axis=0)

    np.random.seed(seed=seed)
    position += cell_size * np.random.uniform(
        low=-0.5,
        high=0.5,
        size=position.shape,
    )

    if vel_offset_fields is not None:
        cell_disp = np.transpose(
            [np.ravel(psi_i) for psi_i in vel_offset_fields]
        )
        displacement = np.repeat(cell_disp, np.ravel(number_field), axis=0)
    else:
        displacement = None

    return position, displacement


def _gen_circsym_whitenoise(num_mesh, seed=None):
    """Generate white noise samples drawn from the circularly-symmetric
    complex normal distribution on a 3-d regular grid.

    Both the real and imaginary parts follow the standard normal
    distribution, so the complex samples have variance 2.

    Parameters
    ----------
    num_mesh : int
        Mesh number per dimension.
    seed : int or None, optional
        Random seed (default is `None`).

    Returns
    -------
    whitenoise : complex :class:`numpy.ndarray`
        Circularly-symmetric Gaussian noise with double unit variance.

    """
    size = (2,) + (num_mesh,)*3
    samples = np.random.RandomState(seed=seed).normal(size=size)

    whitenoise = samples[0] + 1j*samples[1]

    return whitenoise


def _cal_isotropic_power_spectrum(field, boxsize, kmax=None, num_bin=12,
                                  bin_scale='linear'):
    """Calculate the isotropic power spectrum of a random field in
    configuration space.

    Parameters
    ----------
    field : float, array_like
        Random field.
    boxsize : float
        Box size per dimension (in Mpc/h).
    kmax : float or None, optional
        Maximum wavenumber.  If `None` (default), this is set to largest
        wave number the field supports.
    num_bin : int or None, optional
        Number of bins each corresponding to a wavenumber (default is 12).
    bin_scale : {'linear', 'log'}, optional
        Binning in 'linear' (default) or 'log' scale.

    Returns
    -------
    wavenumbers : float, array_like
        Bin wavenumbers.
    powers : float, array_like
        Radially averaged power spectrum at bin wavenumbers.
    mode_count : int, array_like
        Number of modes in bins corresponding to each wavenumber (double
        counting for wave vector parity).

    """
    num_mesh = max(np.array(field).shape)
    vol, num_cell = boxsize**3, num_mesh**3

    k_norm = generate_regular_grid(2*np.pi/boxsize, num_mesh, variable='norm')
    power_arr = vol * np.abs(fftp.fftshift(fftp.fftn(field)))**2 / num_cell**2

    binning_args = (k_norm, power_arr, num_bin, bin_scale)
    powers, wavenumbers, mode_count = _radial_binning(
        *binning_args,
        low_edge=0.,
        high_edge=kmax,
    )

    return wavenumbers, powers, mode_count


def _radial_binning(norm3d, data3d, num_bin, bin_scale, low_edge=None,
                    high_edge=None):
    """Radial binning by coordinate vector norm for 3-d data over a regular
    grid.

    Parameters
    ----------
    norm3d : float, array_like
        Coordinate vector norm grid.
    data3d : float, array_like
        Data array over the same grid.
    num_bin : int
        Number of bins in coordinate norm.
    bin_scale : {'linear', 'log'}
        Binning in 'linear' or 'log' scale.
    low_edge, high_edge : float or None, optional
        Binning range.  If `None` (default), the values are respectively
        set to zero and the largest norm value that the grid supports.

    Returns
    -------
    bin_data : float, array_like
        Bin-averaged data values.
    bin_coord : float, array_like
        Bin-averaged norm values.
    bin_count : int, array_like, optional
        Number of data points in each bin.

    """
    if low_edge is None:
        low_edge = np.min(norm3d)
    if high_edge is None:
        high_edge = np.max(norm3d)

    if bin_scale == 'linear':
        bins = np.linspace(low_edge, high_edge, num_bin+1)
    elif bin_scale == 'log':
        bins = np.logspace(np.log10(low_edge), np.log10(high_edge), num_bin+1)

    bin_count, _ = np.histogram(norm3d.flatten(), bins=bins)
    bin_coord, _ = np.histogram(
        norm3d.flatten(),
        bins=bins,
        weights=norm3d.flatten(),
    )
    bin_data, _ = np.histogram(
        norm3d.flatten(),
        bins=bins,
        weights=data3d.flatten(),
    )

    bin_coord /= bin_count
    bin_data /= bin_count

    return bin_data, bin_coord, bin_count
