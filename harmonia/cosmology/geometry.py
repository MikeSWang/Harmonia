"""
Geometrical calculations (:mod:`~harmonia.cosmology.geometry`)
===========================================================================

Perform geometrical calculations.

.. autosummary::

    fiducial_distance
    redshift_to_distance
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


def redshift_to_distance(z, h0, O0_b, O0_cdm):
    """Compute comoving distance (in Mpc/h) from redshift for a given
    spatially-flat cosmological model with negligible radiation energy
    density and implicit dark energy density.

    Parameters
    ----------
    z : float, array_like
        Redshift.
    h0 : float, array_like
        Reduced Hubble parameter at the current epoch.
    O0_b : float, array_like
        Baryon energy density parameter at the current epoch.
    O0_cdm : float, array_like
        Cold dark matter density parameter at the current epoch.

    Returns
    -------
    float, array_like
        Comoving distance.

    """
    model = cosmology.Cosmology(h=h0, Omega0_b=O0_b, Omega0_cdm=O0_cdm)

    return model.comoving_distance(z)


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
