"""Window effects in measured power spectrum multipoles of simulated
catalogues.

"""
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nbodykit.lab import cosmology

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

from harmonia.collections import harmony
from harmonia.cosmology import fiducial_cosmology
from harmonia.reader import WindowedPowerSpectrum

plt.style.use(harmony)
sns.set(style='ticks', font='serif')


def parse_cli_args():

    cli_parser = ArgumentParser()

    cli_parser.add_argument('--fsky', default="1/3")
    cli_parser.add_argument('--nbar', type=float, default=2.5e-4)
    cli_parser.add_argument('--contrast', type=float, default=10.)
    cli_parser.add_argument('--bias', type=float, default=2.)

    return cli_parser.parse_args()


params = parse_cli_args()

if __name__ == '__main__':

    PATHIN = "./data/input/"
    PATHOUT = "./data/output/"

    REDSHIFT = 0.
    ORDERS = [0, 2, 4]

    fsky = eval(params.fsky)
    nbar = params.nbar
    contrast = params.contrast
    bias = params.bias

    # No window
    matter_power_spectrum = cosmology.LinearPower(
        fiducial_cosmology, redshift=REDSHIFT
    )
    true_power_spectrum = lambda k: bias**2 * matter_power_spectrum(k) \
        + (1 + 1/contrast) / nbar

    # Measurements
    measurements = np.load(
        f"{PATHIN}"
        "window_effects-(fsky=0.33,nbar=0.001,contrast=10.0,boxsize=1000.0,mesh=256,niter=25).npy"
    ).item()

    measurements = {
        'k': measurements['k_win'],
        'modes': measurements['Nk_win'],
        'power_0': measurements['Pk_0_win'],
        'power_2': measurements['Pk_2_win'],
        'power_4': measurements['Pk_4_win'],
    }

    measured_multipoles = {
        key: np.mean(val, axis=0) for key, val in measurements.items()
    }
    measured_multipoles.update({
        'd_' + key: np.std(val, axis=0, ddof=1)
        for key, val in measurements.items()
        if 'power_' in key
    })


    k_universal = measured_multipoles['k']

    # Model
    window_multipoles = np.load(
        f"{PATHIN}window_{{:.2f}}sky.npy".format(fsky)
    ).item()

    windowed_model = WindowedPowerSpectrum(
        redshift=REDSHIFT,
        power_spectrum=matter_power_spectrum,
        cosmo=fiducial_cosmology,
        window=window_multipoles
    )

    windowed_multipoles = windowed_model.convolved_multipoles(
        ORDERS, bias, nbar=nbar, contrast=contrast, wavenumbers=k_universal
    )

    np.save(
        f"{PATHOUT}predicted_multipoles_{{:.2f}}sky.npy".format(fsky),
        windowed_multipoles
    )

    # Comparison
    plt.close('all')
    plt.figure("Window-convolved Power")

    for degree in ORDERS:
        if degree == 0:
            plt.semilogx(
                k_universal,
                true_power_spectrum(k_universal),
                c='k', ls='--', label="isotropic model"
            )
        plt.errorbar(
            k_universal,
            measured_multipoles['power_{}'.format(degree)],
            #measured_multipoles['d_power_{}'.format(degree)],
            label=r"windowed measurements $\ell={}$".format(degree),
            ls='-.'
        )
        plt.semilogx(
            k_universal,
            windowed_multipoles['power_{}'.format(degree)],
            label=r"convolved model $\ell={}$".format(degree)
        )

    plt.legend()

    plt.xlabel(r"$k$ [$h/\textrm{Mpc}$]")
    plt.ylabel(r"$P_\ell(k)$")

    plt.savefig(f"{PATHOUT}convolved_power_{{:.2f}}sky.pdf".format(fsky))
