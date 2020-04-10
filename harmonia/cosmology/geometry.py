"""
Background geometry (:mod:`~harmonia.cosmology.geometry`)
===========================================================================

Compute geometrical quantities in a cosmological background.

.. autosummary::

    differential_AP_distortion

|

"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


def differential_AP_distortion(fiducial_z_to_r, variable_z_to_r,
                               max_redshift=5., log10_num_sample=5):
    """Compute the differential Alcock--Paczynski distortion between a
    fiducial and a cosmological redshift-to-distance conversion as a
    fuction of redshift.

    Parameters
    ----------
    fiducial_z_to_r : callable
        Fiducial redshift-to-distance conversion.
    variable_z_to_r : callable
        Variable redshift-to-distance conversion.
    max_redshift : float
        Maximum redshift to sample for interpolation.
    lg10_num_sample : float, optional
        Base-10 logarithm of the number of redshift points to sample
        the differential distortion as a function of redshift.

    Returns
    -------
    callable
        Differential distortion as a fuction of redshift.

    """
    z_samples = np.linspace(0, max_redshift, 10 ** log10_num_sample)

    r_tilde = fiducial_z_to_r(z_samples)
    r = variable_z_to_r(z_samples)

    with np.errstate(divide='ignore'):
        dr_tilde_over_dr = np.gradient(r_tilde) / np.gradient(r)

    # Filter out where possibly the gradient of r(z) is close to zero.
    valid_samples = np.isfinite(dr_tilde_over_dr)

    z_samples = z_samples[valid_samples]
    dr_tilde_over_dr = dr_tilde_over_dr[valid_samples]

    return Spline(z_samples, dr_tilde_over_dr, check_finite=False)
