"""
Coordinates (:mod:`~harmonia.surveyor.coordinates`)
===========================================================================

Handle the survey coordinate systems.

.. autosummary::

    cartesian_to_spherical
    spherical_to_cartesian
    sky_to_spherical
    spherical_to_sky
    to_box_coords

|

"""
import warnings
from functools import wraps

import numpy as np


def cartesian_to_spherical(cartesian_coords):
    r"""Convert 3-d Cartesian coordinate arrays to spherical coordinate
    arrays.

    The coordinate transformation is given by

    .. math::

        r = \sqrt{x^2 + y^2 + z^2} \,, \quad
        \theta = \arccos(z/r) \,, \quad
        \phi = \arctan(y/x) \,,

    where the image of :math:`\arccos` is :math:`[0, \pi]`, and
    :math:`\arctan` has an extended image set :math:`[0, 2\pi]`.

    Parameters
    ----------
    cartesian_coords : float, array_like
        Cartesian coordinates.

    Returns
    -------
    spherical_coords : float :class:`numpy.ndarray`
        Spherical coordinates.

    """
    c_coords = np.atleast_2d(cartesian_coords)
    if not _is_coord_3d(c_coords):
        raise ValueError("`cartesian_coords` is not 3-d.")

    spherical_coords = np.zeros(c_coords.shape)
    spherical_coords[:, 0] = np.linalg.norm(c_coords, axis=1)
    spherical_coords[:, 1] = np.arccos(c_coords[:, 2] / spherical_coords[:, 0])
    spherical_coords[:, 2] = np.mod(
        np.arctan2(c_coords[:, 1], c_coords[:, 0]), 2*np.pi
    )

    return spherical_coords


def spherical_to_cartesian(spherical_coords):
    r"""Convert 3-d spherical coordinate arrays to Cartesian coordinate
    arrays.

    The coordinate transformation is given by

    .. math::

        x = r \sin\theta \cos\phi \,, \quad
        y = r \sin\theta \sin\phi \,, \quad
        z = r \cos\theta \,.

    Parameters
    ----------
    spherical_coords : float, array_like
        Spherical coordinates.

    Returns
    -------
    cartesian_coords : float :class:`numpy.ndarray`
        Cartesian coordinates.

    """
    s_coords = np.atleast_2d(spherical_coords)
    if not _is_coord_3d(s_coords):
        raise ValueError("`spherical_coords` is not 3-d.")

    cartesian_coords = np.zeros(s_coords.shape)
    cartesian_coords[:, 0] = np.sin(s_coords[:, 1]) * np.cos(s_coords[:, 2])
    cartesian_coords[:, 1] = np.sin(s_coords[:, 1]) * np.sin(s_coords[:, 2])
    cartesian_coords[:, 2] = np.cos(s_coords[:, 1])

    cartesian_coords *= (s_coords[:, 0])[:, None]

    return cartesian_coords


def sky_to_spherical(sky_coords, z_to_r=None):
    r"""Convert 3-d (or 2-d) sky coordinate arrays (Z, DEC, RA)
    (or (DEC, RA)) to spherical coordinate arrays.

    The spherical surface coordinate transformation is given by

    .. math::

        \theta = \pi/180 * (90 - \delta) \,, \quad
        \phi = \pi/180 * \alpha \,.

    where :math:`\delta` is the declination (DEC) and :math:`\alpha`
    the right ascension (RA) both given in degrees.

    The radial coordinate transformation from redshifts is given by
    :meth:`cosmo.comoving_distance` if the input coordinates are 3-d
    and a redshift-to-distance conversion function is provided.

    Parameters
    ----------
    sky_coords : float, array_like
        Sky coordinates (2-d or 3-d).
    z_to_r : callable or None, optional
        Redshift-to-distance conversion (default is `None`).

    Returns
    -------
    spherical_coords : float :class:`numpy.ndarray`
        Spherical surface coordinates.

    """
    sky_coords = np.atleast_2d(sky_coords)

    if _is_coord_3d(sky_coords):
        if not callable(z_to_r):
            raise TypeError(
                "`z_to_r` must be provided for radial "
                "coordinate transformation."
            )

        spherical_coords = np.zeros(sky_coords.shape)
        spherical_coords[:, 0] = z_to_r(sky_coords[:, 0])
        spherical_coords[:, 1] = np.deg2rad(90 - sky_coords[:, 1])
        spherical_coords[:, 2] = np.deg2rad(sky_coords[:, 2])

        return spherical_coords

    if _is_coord_2d(sky_coords):

        spherical_coords = np.zeros(sky_coords.shape)
        spherical_coords[:, 0] = np.deg2rad(90 - sky_coords[:, 0])
        spherical_coords[:, 1] = np.deg2rad(sky_coords[:, 1])

        return spherical_coords

    raise ValueError("Check dimensions of input `sky_coords`.")


def spherical_to_sky(spherical_coords, z_from_r=None):
    r"""Convert 3-d spherical coordinate arrays to sky coordinate arrays.

    The spherical surface coordinate transformation is given by

    .. math::

        \delta = 90 - 180/\pi * \theta \,, \quad
        \alpha = 180/\pi * \phi \,.

    where :math:`\delta` is the declination (DEC) and :math:`\alpha`
    the right ascension (RA) both given in degrees.

    The radial coordinate transformation is given by a
    distance-to-redshift conversion if it provided.

    Parameters
    ----------
    sky_coords : float, array_like
        Sky coordinates.
    z_from_r : callable or None, optional
        Distance-to-redshift conversion (default is `None`).

    Returns
    -------
    sky_coords : float :class:`numpy.ndarray`
        Sky coordinates.

    """
    spherical_coords = np.atleast_2d(spherical_coords)
    if not _is_coord_3d(spherical_coords):
        raise ValueError("`spherical_coords` is not 3-d.")

    sky_coords = np.zeros(spherical_coords.shape)

    sky_coords[:, 1] = 90 - np.rad2deg(spherical_coords[:, 1])
    sky_coords[:, 2] = np.rad2deg(spherical_coords[:, 2])

    if z_from_r is None:

        sky_coords = np.delete(sky_coords, 0, axis=1)

        return sky_coords

    sky_coords[:, 0] = z_from_r(spherical_coords[:, 0])

    return sky_coords


def _is_coord_2d(coords):
    return np.size(coords, axis=-1) == 2


def _is_coord_3d(coords):
    return np.size(coords, axis=-1) == 3


# pylint: disable=unused-argument
def to_box_coords(native_coord_system, box_centre=None,
                  conversion_kwargs=None):
    """Convert a function defined for a native 3-d curvilinear coordinate
    system to an equivalent function defined for a box in
    Cartesian coordinates.

    Parameters
    ----------
    native_coord_system : {'null', 'spherical', 'sky'}, optional
        Native coordinate system of the function.  If 'cartesian' (with
        the implicit assumption that the origin is at the box centre),
        the function is unconverted unless `box_shift` is also provided.
    box_centre : float, array_like or Nonw
        If provided, translate the box coordinates so that the (positive)
        `box_centre` is moved to(0, 0, 0).  This is needed to when the
        wrapped function needs to accept box coordinates with the origin
        placed at its corner whilst the `native_coord_system` has the
        origin at the centre of the box (e.g. 'spherical').
    conversion_kwargs : dict or None, optional
        Additional parameters to use in conversion, e.g. `z_from_r`
        if `native_coords` is 'sky' (default is `None`).

    Returns
    -------
    callable
        The original function now accepting Cartesian coordinates.

    """
    def decorator(func):
        @wraps(func)
        def wrapper(*coord_arrs, **kwargs):

            nonlocal native_coord_system

            # Offer the option to do nothing.
            if box_centre is None and native_coord_system == 'null':
                return func(*coord_arrs, **kwargs)

            native_coord_system = 'cartesian' \
                if native_coord_system == 'null' \
                else native_coord_system

            if len(coord_arrs) == 1:
                unpacked = False
            elif len(coord_arrs) == 3:
                unpacked = True
            else:
                raise ValueError(
                    "Wrapped coordinate function must only have positional "
                    "arguments that are 3-d coordinate arrays."
                )

            if unpacked:
                coords = np.column_stack(coord_arrs)
            else:
                coords, = coord_arrs

            if box_centre is None:
                cart_coords = np.atleast_2d(coords)
            else:
                if np.any(np.sign(box_centre) < 0):
                    warnings.warn(
                        "`box_shift` is usually positive to ensure "
                        "the origin of the Cartesian coordinates "
                        "is at a corner of the box."
                    )
                cart_coords = np.subtract(coords, box_centre)

            if native_coord_system == 'cartesian':
                native_coords = cart_coords
            elif native_coord_system == 'spherical':
                native_coords = cartesian_to_spherical(cart_coords)
            elif native_coord_system == 'sky':
                native_coords = spherical_to_sky(
                    cartesian_to_spherical(cart_coords), **conversion_kwargs
                )
            else:
                raise ValueError(
                    "Unsupported `native_coord_system`: "
                    f"{native_coord_system}."
                )

            if unpacked:
                native_coord_arrs = np.hsplit(native_coords, 3)
            else:
                native_coord_arrs = (native_coords,)

            return func(*native_coord_arrs, **kwargs)
        return wrapper
    return decorator
