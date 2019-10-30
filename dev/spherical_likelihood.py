"""Spherical likelihood constructed from data realisation and covariance
modelling.

"""
import numpy as np

from complex_normal import complex_normal_log_pdf

_OVERFLOW_DOWNSCALE = 10**4


def parametrised_covariance(two_point_model, pivot, nbar, b_const, f_nl,
                            independence=False, diag=False):
    r"""Parametrised covariance matrix given the 2-point function base
    model and the pivot axis for data vectorisation of transformed fields.

    Parameters
    ----------
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function base model.
    pivot : {'natural', 'transposed', 'spectral'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    f_nl : float
        Local primordial non-Gaussianity.
    b_const : float
        Consparam_covar_kwargstant linear bias.
    independence : bool, optional
        If `True` (default is `False`), independence amongst Fourier modes
        is assumed and the diagonal covariance matrix is computed without
        coupling coefficients.
    diag : bool, optional
        If `True` (default is `False`), only the diagonal elements of the
        covariance matrix are computed by summation against of coupling
        coefficients.

    Returns
    -------
    covariance : complex or float :class:`numpy.ndarray`
        Parametrised covariance matrix.

    See Also
    --------
    :class:`~harmonia.reader.spherical_model.TwoPointFunction`

    """
    if independence:
        variance = two_point_model.mode_variance(
            pivot,
            nbar=nbar,
            b_const=b_const,
            f_nl=f_nl
        )
        covariance = np.diag(variance)
    else:
        covariance = two_point_model.two_point_covariance(
            pivot,
            diag=diag,
            nbar=nbar,
            b_const=b_const,
            f_nl=f_nl
        )

    return covariance


def spherical_map_likelihood(param_points, param_name, spherical_data,
                             two_point_model, pivot, nbar, bias=None,
                             f_nl=None, breakdown=False, remove_degrees=(),
                             mode_indices=None, **param_covar_kwargs):
    """Evaluate the spherical map logarithmic likelihood.

    Parameters
    ----------
    param_points : float, array_like
        Parameter sampling values.
    param_name : {'f_nl', 'bias'}
        Name of the parameter to be sampled, either ``'f_nl'`` for the
        local primordial non-Gaussianity or ``'bias'`` for the
        scale-independent linear bias.
    spherical_data : :class:`~harmonia.algorithms.morph.SphericalArray`
        Spherical data array of the transformed field.
    two_point_model : :class:`~.spherical_model.TwoPointFunction`
        2-point function base model.
    pivot : {'natural', 'transposed', 'spectral', 'root', 'scale'}
        Pivot axis for unpacking indexed data into a 1-d vector.
    nbar : float
        Mean particle number density (in cubic h/Mpc).
    bias : float or None, optional
        Constant linear bias of the tracer particles at the current
        epoch.  Cannot be `None` (default) if `param_name` is ``'f_nl'``.
    f_nl : float or None, optional
        Local primordial non-Gaussianity.  Cannot be `None` (default) if
        `param_name` is ``'bias'``.

    Other parameters
    ----------------
    remove_degrees : int, array_like, optional
        If not an empty tuple (default), modes whose spherical degree is an
        element are removed from the data vector and parametrised
        covariance.
    mode_indices : list of (int, int, int) or None, optional
        Mode indices of the spherical data.  Cannot be `None` (default) if
        `remove_degrees` is not `None`.
    breakdown : bool, optional
        If `True` (default is `False`), the contribution from each data
        vector is broken down but the covariance matrix must be diagonal.
    **param_covar_kwargs
        Keyword arguments `independence` and `diag` for
        :func:`~.parametrised_covariance`.

    Returns
    -------
    log_likelihood : float :class:`numpy.ndarray`
        Logarithmic likelihood evaluated at the parameter sampling points.

    Raises
    ------
    ValueError
        If `bias` is `None` when `param_name` is ``'f_nl'`` or `f_nl` is
        `None` when `param_name` is ``'bias'``, or if `param_name` neither.

    """
    data_vector = spherical_data.unfold(pivot, return_only='data')
    if remove_degrees:
        if mode_indices is None:
            raise ValueError(
                "`mode_indices` must be provided "
                "if `remove_degrees` is non-empty. "
            )
        excluded_deg = np.fromiter(
            map(lambda index: index[0] in remove_degrees, mode_indices),
            dtype=bool
        )
        data_vector = data_vector[~excluded_deg]

    param_covar_args = (two_point_model, pivot, nbar)

    if breakdown:
        log_likelihood = np.zeros((len(param_points), len(data_vector)))
    else:
        log_likelihood = np.zeros_like(param_points)
    for idx, param in enumerate(param_points):
        if param_name == 'f_nl':
            if bias is None:
                raise ValueError(
                    "`bias` value must be provided "
                    "for sampling non-Gaussnaity. "
                )
            sample_covar = parametrised_covariance(
                *param_covar_args,
                bias,
                param,
                **param_covar_kwargs
            )
        elif param_name == 'bias':
            if f_nl is None:
                raise ValueError(
                    "`f_nl` value must be provided "
                    "for sampling scale-independent bias. "
                )
            sample_covar = parametrised_covariance(
                *param_covar_args,
                param,
                f_nl,
                **param_covar_kwargs
            )
        else:
            raise ValueError(
                f"Invalid `param_name` for sampling: {param_name}. "
            )

        if remove_degrees:
            sample_covar = sample_covar[:, ~excluded_deg][~excluded_deg, :]

        log_likelihood[idx] = complex_normal_log_pdf(
            data_vector,
            sample_covar,
            overflow_rescale=_OVERFLOW_DOWNSCALE,
            breakdown=breakdown
        )

    return log_likelihood
