"""
lognormal_transform (:mod:`~harmonia.algorithms.lognormal_transform`)
===============================================================================

Log-normal transform a configuration-space random field on a regular grid.

"""
import numpy as np


def perform_lognormal_transform(gaussian_random_field):
    """Perform log-normal transform of a Gaussian random field on a regular
    grid,

    Parameters
    ----------
    gaussian_random_field : :class:`numpy.ndarray` of float
        Gaussian random field in configuration space.

    Returns
    -------
    lognormal_field : :class:`numpy.ndarray` of float
        Log-normal random field in configuration space.

    """
    field_var = np.var(gaussian_random_field)

    lognormal_field = np.exp(-field_var + gaussian_random_field) - 1

    return lognormal_field


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from nbodykit.lab import cosmology as cosmo

    import sys  #
    sys.path.insert(0, "../")  #

    from harmonia.collections import harmony
    from gaussian_random_field import (
        gen_gaussian_random_field, compute_isotropic_power_spectrum
        )  #
    # from harmonia.algorithms import (
    #     gen_gaussian_random_field, compute_isotropic_power_spectrum
    #     )

    boxside = 5000.
    nmesh = 256

    Plin = cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')

    field = gen_gaussian_random_field(boxside, nmesh, Plin, seed=81)
    field = perform_lognormal_transform(field)

    k, Pk = compute_isotropic_power_spectrum(field, boxside, kmax=0.10)

    plt.style.use(harmony)
    plt.close('all')
    plt.figure('Log-normal random field')
    plt.loglog(k, Pk, label='recovered power')
    plt.loglog(k, Plin(k), '--', label='model spectrum')
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.legend()
