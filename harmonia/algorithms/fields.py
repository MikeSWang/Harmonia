"""
Random fields (:mod:`~harmonia.algorithms.fields`)
===========================================================================

Generate random fields on 3-d regular grids from an input power spectrum in
a cubic box, and perform biasing, threshold clipping, log-normal
transformation and discrete Poisson sampling of fields with particle
population.

.. warning::

    Experimental module in replacement of ``nbodykit`` catalogue makers.
    Use with caution.

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
from __future__ import division

import warnings

import numpy as np
import scipy.fftpack as fftp


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
        The grid variable to be returned: ``'coords'`` for the grid
        coordinates, ``'norm'`` (default) of the grid coordinates, or
        ``'both'`` in that order.

    Returns
    -------
    grid_coords : list of float :class:`numpy.ndarray`
        Grid coordinate arrays for each dimension.  Returned if `variable`
        is ``'coords'`` or ``'both'``.
    grid_norm : float :class:`numpy.ndarray`
        Grid coordinate norm array.  Returned if `variable` is ``'norm'``
        or ``'both'``.

    Raises
    ------
    ValueError
        If `variable` does not correspond to any of the following:
        ``'coords'``, ``'norm'`` or ``'both'``.

    """
    indices = np.indices((num_mesh,) * 3)
    origin = np.array([(num_mesh-1)/2] * 3)

    grid_coords = [
        cell_size * (index - centre)
        for index, centre in zip(indices, origin)
    ]
    grid_norm = np.sqrt(np.sum([coord**2 for coord in grid_coords], axis=0))

    if variable.lower().startswith('c'):
        return grid_coords
    if variable.lower().startswith('n'):
        return grid_norm
    if variable.lower().startswith('b'):
        return grid_coords, grid_norm
    raise ValueError(f"Unknown grid `variable`: {variable}. ")


def generate_gaussian_random_field(boxsize, num_mesh, power_spectrum, bias=1.,
                                   clip=True, return_disp=False, seed=None):
    r"""Generate a Gaussian random field corresponding to the density
    contrast :math:`\delta(\mathbf{r})` in configuration space with desired
    input power spectrum and bias, and optionally derive a random vector
    field without bias corresponding to the velocity displacement
    :math:`\boldsymbol{\Psi}(\mathbf{r})`.

    In Fourier space, the displacement field is related to the density
    contrast field by

    .. math::

        \boldsymbol{\Psi}(\mathbf{k}) = \mathrm{i} \frac{\mathbf{k}}{k^2}
        \delta(\mathbf{k}) \,,

    and to the velocity field by

    .. math::

        \mathbf{v}(\mathbf{k}) = Haf \Psi(\mathbf{k}) \,.

    Parameters
    ----------
    boxsize : float
        Box size per dimension (in Mpc/:math:`h`).
    num_mesh : int
        Mesh number per dimension.
    power_spectrum : callable
        Desired power spectrum (in cubic Mpc/:math:`h`).
    bias : float or callable, optional
        Bias of the density contrast field (default is 1.), as a constant
        or a function of the Fourier wavenumber.
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
    displacement : list of float :class:`numpy.ndarray` or None
        Gaussian random fields of velocity displacement in configuration
        space for each dimension.  Returned as `None` unless `return_disp`
        is `True`.

    Raises
    ------
    ValueError
        If `bias` is neither a positive float nor a callable function.

    """
    vol, num_cell = boxsize**3, num_mesh**3
    k_vec, k_norm = generate_regular_grid(
        2*np.pi/boxsize, num_mesh, variable='both'
    )

    whitenoise = _gen_circsym_whitenoise(num_mesh, seed=seed)
    amplitude = np.sqrt(power_spectrum(k_norm) / vol)
    fourier_field = amplitude * whitenoise

    try:
        bias = float(bias)
    except TypeError:
        if callable(bias):
            bias_k = bias(k_norm)
            overdensity = num_cell * np.real(
                fftp.ifftn(fftp.fftshift(bias_k * fourier_field))
            )
        else:
            raise ValueError(f"Invalid `bias` parameter: {bias}. ")
    else:
        if bias <= 0:
            raise ValueError("`bias` parameter must be positive. ")
        overdensity = num_cell * np.real(
            fftp.ifftn(fftp.fftshift(bias * fourier_field))
        )

    if clip:
        overdensity = threshold_clip(overdensity, threshold=-1.)

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
    contrast :math:`\delta(\mathbf{r})` in configuration space with desired
    input power spectrum and bias, and optionally derive a random vector
    field without bias corresponding to the velocity displacement
    :math:`\boldsymbol{\Psi}(\mathbf{r})` which fullfils the continuity
    equation.

    Parameters
    ----------
    boxsize : float
        Box size per dimension (in Mpc/:math:`h`).
    num_mesh : int
        Mesh number per dimension.
    power_spectrum : callable
        Desired power spectrum (in cubic Mpc/:math:`h`).
    bias : float or callable, optional
        Bias of the density contrast field (default is 1.), as a constant
        or a function of the Fourier wavenumber.
    return_disp : bool, optional
        If `True` (default is `False`), also return the velocity
        displacement field for each dimension that is not `None`.
    seed : int or None, optional
        Random seed for the field(s) (default is `None`).

    Returns
    -------
    overdensity : float :class:`numpy.ndarray`
        Gaussian random field of density contrast in configuration space.
    displacement : list of float :class:`numpy.ndarray` or None
        Gaussian random fields of velocity displacement in configuration
        space for each dimension.  Returned as `None` if `return_disp`
        is `False`.

    Raises
    ------
    ValueError
        If `bias` is neither a positive float nor a callable function.

    See Also
    --------
    :func:`generate_gaussian_random_field`

    """
    vol, num_cell = boxsize**3, num_mesh**3
    k_vec, k_norm = generate_regular_grid(
        2*np.pi/boxsize, num_mesh, variable='both'
    )

    pk = power_spectrum(k_norm) / vol

    try:
        bias = float(bias)
    except TypeError:
        if callable(bias):
            pk_target = bias(k_norm)**2 * pk
        else:
            raise ValueError(f"Invalid `bias` parameter: {bias}. ")
    else:
        if bias <= 0:
            raise ValueError("`bias` parameter must be positive. ")
        pk_target = bias**2 * pk

    xi_target = num_cell * np.real(fftp.ifftn(fftp.fftshift(pk_target)))
    xi_generation = lognormal_transform(xi_target, 'correlation')
    pk_generation = fftp.fftshift(fftp.fftn(xi_generation)) / num_cell

    whitenoise = _gen_circsym_whitenoise(num_mesh, seed=seed)

    fourier_field_generation = np.sqrt(pk_generation) * whitenoise
    overdensity_generation = num_cell * np.real(
        fftp.ifftn(fftp.fftshift(fourier_field_generation))
    )

    overdensity = lognormal_transform(overdensity_generation, 'field')

    if return_disp:
        fourier_field = np.sqrt(pk) * whitenoise
        fourier_disp = [1j * ki / k_norm**2 * fourier_field for ki in k_vec]
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
    density_contrast = np.array(density_contrast)

    clipping_mask = density_contrast < threshold
    clipping_ratio = np.sum(clipping_mask) / np.size(clipping_mask)
    if clipping_ratio > 0.005:
        warnings.warn(
            "{:.2g}% of field values are clipped. ".format(100*clipping_ratio),
            RuntimeWarning
        )

    density_contrast[clipping_mask] = -1.

    return density_contrast


def lognormal_transform(obj, obj_type):
    r"""Perform log-normal transform of a field or its 2-point correlation
    function values in configuration space.

    Parameters
    ----------
    obj : array_like or callable
        The object to be transformed.
    obj_type : {'field', 'correlation'}
        Type of the object to be transformed: ``'field'`` values or 2-point
        ``'correlation'`` function values.

    Returns
    -------
    float :class:`numpy.ndarray` or callable
        Log-normal transformed object.

    Raises
    ------
    TypeError
        If `obj` type is neither array_like nor callable.
    ValueError
        If `obj_type` does not correspond to either ``'field'`` or
        ``'correlation'``.

    """
    if obj_type.lower().startswith('f'):

        density_field = np.exp(obj)
        density_mean = np.mean(density_field)
        transformed_field = density_field / density_mean - 1

        return transformed_field

    if obj_type.lower().startswith('c'):
        try:
            return np.log(1 + obj)
        except TypeError:
            if callable(obj):
                return lambda r: np.log(1 + obj(r))
            raise TypeError(f"Invalid `obj` type: {type(obj)}. ")

    raise ValueError(f"Invalid `obj_type` value: {obj_type}. ")


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
    density_contrast = np.array(density_contrast)
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
                       velocity_offset_fields=None, seed=None):
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
    velocity_offset_fields : list of float, array_like, optional
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
    sampled_field = np.array(sampled_field)
    if len(set(sampled_field.shape)) > 1:
        raise ValueError("`field` is not a regular grid. ")

    num_mesh = max(sampled_field.shape)
    cell_size = boxsize / num_mesh

    grid_coords = generate_regular_grid(cell_size, num_mesh, variable='coords')
    number_field = np.int64(
        0.5 + (1 + sampled_field) * mean_density * cell_size**3
    )

    cell_pos = np.transpose([np.ravel(coords) for coords in grid_coords])
    position = np.repeat(cell_pos, np.ravel(number_field), axis=0)

    np.random.seed(seed=seed)
    position += cell_size * np.random.uniform(
        low=-0.5, high=0.5, size=position.shape
    )

    displacement = None
    if velocity_offset_fields is not None:
        cell_disp = np.transpose(
            [np.ravel(psi_i) for psi_i in velocity_offset_fields]
        )
        displacement = np.repeat(cell_disp, np.ravel(number_field), axis=0)

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

    whitenoise = samples[0] + 1j * samples[1]

    return whitenoise


def _cal_isotropic_power_spectrum(field, boxsize, kmax=None, num_bin=12,
                                  scaling='linear'):
    """Calculate the isotropic power spectrum of a random field in
    configuration space.

    Parameters
    ----------
    field : float, array_like
        Random field.
    boxsize : float
        Box size per dimension (in Mpc/:math:`h`).
    kmax : float or None, optional
        Maximum wavenumber.  If `None` (default), this is set to largest
        wave number the field supports.
    num_bin : int or None, optional
        Number of bins each corresponding to a wavenumber (default is 12).
    scaling : {'linear', 'log'}, optional
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
    power_k = vol * np.abs(fftp.fftshift(fftp.fftn(field)))**2 / num_cell**2

    binning_args = (k_norm, power_k, num_bin, scaling)
    powers, wavenumbers, mode_count = _radial_binning(
        *binning_args, low_edge=0., high_edge=kmax
    )

    return wavenumbers, powers, mode_count


def _radial_binning(grid_norm, grid_data, num_bin, scaling, low_edge=None,
                    high_edge=None):
    """Radial binning by 3-d coordinate vector norm for data over a regular
    cubic grid.

    Parameters
    ----------
    grid_norm : float, array_like
        Coordinate vector norm grid.
    grid_data : float, array_like
        Data array over the same grid.
    num_bin : int
        Number of bins in coordinate norm.
    scaling : {'linear', 'log'}
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
        low_edge = np.min(grid_norm)
    if high_edge is None:
        high_edge = np.max(grid_norm)

    if scaling == 'linear':
        bins = np.linspace(low_edge, high_edge, num_bin+1)
    elif scaling == 'log':
        bins = np.logspace(np.log10(low_edge), np.log10(high_edge), num_bin+1)

    bin_count, _ = np.histogram(grid_norm.flatten(), bins=bins)

    bin_coord, _ = np.histogram(
        grid_norm.flatten(),
        bins=bins,
        weights=grid_norm.flatten()
    )
    bin_coord /= bin_count

    bin_data, _ = np.histogram(
        grid_norm.flatten(),
        bins=bins,
        weights=grid_data.flatten()
    )
    bin_data /= bin_count

    return bin_data, bin_coord, bin_count
