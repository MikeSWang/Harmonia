"""
Geometrical calculations (:mod:`~harmonia.cosmology.geometry`)
===============================================================================

Perform geometrical calculations.

.. autosummary::

    fiducial_distance
    redshift_to_distance

|

"""
from nbodykit import cosmology as cosmo


def fiducial_distance(z):
    """Compute comoving distance from redshift in fiducial cosmology.

    Parameters
    ----------
    z : float, array_like
        Redshift.

    Returns
    -------
    float, array_like
        Fiducial comoving distance (in Mpc/h).

    """
    return cosmo.Planck15.comoving_distance(z)


def redshift_to_distance(z, h0, O0_b, O0_cdm):
    """Compute comoving distance from redshift for a given spatially-flat
    cosmological model with negligible radiation energy density and implicit
    dark energy density.

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
        Comoving distance (in Mpc/h).

    """
    model = cosmo.Cosmology(h=h0, Omega0_b=O0_b, Omega0_cdm=O0_cdm)

    return model.comoving_distance(z)
