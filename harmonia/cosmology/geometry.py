"""
Geometrical calculations (:mod:`~harmonia.cosmology.geometry`)
===========================================================================

Perform geometrical calculations.

.. autosummary::

    fiducial_distance
    differential_AP_distortion

|

"""
from nbodykit import cosmology

_fiducial_cosmology = cosmology.Planck15


def fiducial_distance(z):
    """Compute comoving distance (in Mpc/h) from redshift in fiducial
    cosmology.

    Parameters
    ----------
    z : float, array_like
        Redshift.

    Returns
    -------
    float, array_like
        Comoving distance.

    """
    return _fiducial_cosmology.comoving_distance(z)


def differential_AP_distortion(chi_z_fiducial, chi_z_cosmological):
    """Compute the Alcock--Paczynski differential distortion between a
    fiducial and a cosmological redshift-to-distance conversion as a
    fuction of redshift.

    Parameters
    ----------
    chi_z_fiducial : callable
        Fiducial redshift-to-distance conversion as a fuction of redshift.
    chi_z_cosmological : callable
        Cosmological redshift-to-distance conversion as a fuction of
        redshift.

    Returns
    -------
    differential_distortion : callable
        Differential distortion as a fuction of redshift.


    .. todo:: Not yet implemented.

    """
    raise NotImplementedError
