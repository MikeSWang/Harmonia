"""Window effects in measured power spectrum multipoles of simulated
catalogues.

"""
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

_cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))

from harmonia.collections import confirm_directory_path, harmony
from harmonia.reader import WindowedPowerSpectrum

plt.style.use(harmony)
sns.set(style='ticks', font='serif')

PK_FILENAME = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"


def parse_cli_args():

    cli_parser = ArgumentParser()

    cli_parser.add_argument('--fsky', type=float, default=1.)
    cli_parser.add_argument('--bias', type=float, default=2.33)
    cli_parser.add_argument('--nbar', type=float, default=2.4883e-4)
    cli_parser.add_argument('--contrast', type=float, default=20.)

    return cli_parser.parse_args()


def shift_x_coord(degree):

    if degree == 0:
        return 1.
    if degree == 2:
        return 0.996
    if degree == 4:
        return 1.004


def shift_y_coord(degree):

    if degree == 0:
        return 0.
    if degree == 2:
        return 250.
    if degree == 4:
        return -250.


params = parse_cli_args()

if __name__ == '__main__':

    PATHIN = "./data/input/"
    PATHOUT = "./data/output/"
    SCRIPT_NAME = "window_convolution"
    confirm_directory_path(f"{PATHOUT}{SCRIPT_NAME}/")

    REDSHIFT = 0.
    ORDERS = [0, 2, 4]

    fsky = params.fsky
    bias = params.bias
    nbar = params.nbar
    contrast = params.contrast

    # Underlying power spectrum.
    matter_power_spectrum = Spline(
        *np.loadtxt(f"{PATHIN}cosmology/{PK_FILENAME}.txt", unpack=True), k=1
    )

    # Window function.
    mask_multipoles = np.load(
        f"{PATHOUT}window_multipoles/mask_multipoles-{{:.2f}}sky-70pad.npy"
        .format(fsky)
    ).item()
    try:
        window_multipoles = np.load(
            f"{PATHOUT}window_multipoles/"
            "window_multipoles-{:.2f}sky-70pad.npy"
            .format(fsky)
        ).item()
    except FileNotFoundError:
        window_multipoles = None

    # Measurements
    measurements = np.load(
        f"{PATHOUT}window_effects/windowed_measurements"
        "-(fsky={:.2f},contrast={:.1f},mesh=256).npy"
        .format(fsky, contrast)
    ).item()

    measured_multipoles = {
        'k': np.mean(np.concatenate(measurements['k']), axis=0),
        'modes': np.sum(np.concatenate(measurements['modes']), axis=0),
    }

    measured_multipoles.update({
        key: np.mean(np.concatenate(val), axis=0)
        for key, val in measurements.items()
        if 'power_' in key
    })
    measured_multipoles.update({
        'd' + key: np.std(np.concatenate(val), axis=0, ddof=1)
        for key, val in measurements.items()
        if 'power_' in key
    })

    # Model
    k_data = measured_multipoles['k']
    k_model = np.linspace(np.min(k_data), np.max(k_data), num=1024)

    windowed_model = WindowedPowerSpectrum(
        redshift=REDSHIFT,
        power_spectrum=matter_power_spectrum,
        mask_multipoles=mask_multipoles,
        window_multipoles=window_multipoles
    )

    windowed_multipoles = windowed_model.convolved_multipoles(
        ORDERS, bias, nbar=nbar, contrast=contrast, wavenumbers=k_model
    )

    # Comparison
    plt.close('all')
    plt.figure("Window-convolved Power", figsize=(7, 5))

    plt.semilogx(
        k_model, bias**2 * matter_power_spectrum(k_model) + 1/nbar,
        c='k', ls=':', label="isotropic model"
    )
    for ell in ORDERS:
        measurement_plot = plt.errorbar(
            measured_multipoles['k'] * shift_x_coord(ell),
            measured_multipoles['power_{}'.format(ell)],
            measured_multipoles['dpower_{}'.format(ell)],
            fmt='o', markersize=3., capsize=0., alpha=2/3,
            label=r"measurements $\ell={}$".format(ell)
        )
        plt.semilogx(
            k_model,
            windowed_multipoles['power_{}'.format(ell)] + shift_y_coord(ell),
            color=measurement_plot[0].get_color(),
            linestyle='--', label=r"model $\ell={}$".format(ell)
        )

    plt.legend()
    plt.title(r"$f_\mathrm{{sky}} = {:.2f}$".format(fsky))

    plt.xlim(0.008, 0.1)
    plt.ylim(-3*10**4, .9*10**5)
    plt.xlabel(r"$k$ [$h/\textrm{Mpc}$]")
    plt.ylabel(r"$P_\ell(k)$")
    plt.subplots_adjust(hspace=0., wspace=0.)
