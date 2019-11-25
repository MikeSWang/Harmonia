"""Calculate the survey window function.

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from harmonia.mapper import SurveyWindow
from harmonia.collections import (
    cartesian_to_spherical,
    harmony,
    spherical_indicator,
)

plt.style.use(harmony)
sns.set(style='ticks', font='serif')


def sky_mask(cartesian_position, fraction):
    """Define the sky mask.

    Parameters
    ----------
    cartesian_position : float array_like
        Cartesian positions.
    fraction : float
        Sky fraction.

    Returns
    -------
    mask: bool :class:`numpy.ndarray`
        Sky mask.

    """
    spherical_position = cartesian_to_spherical(cartesian_position)

    mask = spherical_position[..., -1] <= fraction * (2*np.pi)

    return mask


def synthesise():
    """Synthesise a high-density random catalogue with the surveywindow.

    Returns
    -------
    window: :class:`harmonia.mapper.window_synthesis.SurveyWindow`
        Survey window.

    """
    window = SurveyWindow(
        mask=lambda pos:\
            spherical_indicator(pos, BOXSIZE/2) * sky_mask(pos, SKY_FRAC)
    )
    window.synthesise(NBAR, BOXSIZE, padding=PADDING)

    return window


def determine(window):
    """Determine window multipoles from the synthetic catalogue.

    Parameters
    ----------
    window: :class:`harmonia.mapper.window_synthesis.SurveyWindow`
        Survey window.

    Returns
    -------
    xi_ell, pk_ell: dict
        Survey window correlation and power multipoles.

    """
    xi_ell = window.correlation_function_multipoles([0, 2, 4, 6, 8])
    pk_ell = window.power_multipoles

    np.save(f"{PATHOUT}window-{{:.2f}}.npy".format(SKY_FRAC), xi_ell)

    return xi_ell, pk_ell


if __name__ == '__main__':

    PATHOUT = "./data/output"

    SKY_FRAC = 1/3
    NBAR = 5e-2
    BOXSIZE = 1000.
    PADDING = 80.

    window = synthesise()
    xi_ell, pk_ell = determine(window)

    plt.close('all')

    plt.figure()

    plt.semilogx(pk_ell['k'], pk_ell['power_0'], label='monopole')
    plt.semilogx(pk_ell['k'], pk_ell['power_2'], label='quadrupole')
    plt.semilogx(pk_ell['k'], pk_ell['power_4'], label='hexadecapole')

    plt.xlabel(r"$k$")
    plt.ylabel(r"$Q_\ell(k)$")
    plt.legend()

    plt.savefig(f"{PATHOUT}window_power_{{:.2f}}sky.pdf".format(SKY_FRAC))

    plt.figure()

    plt.semilogx(xi_ell['s'], xi_ell['correlation_0'], label='monopole')
    plt.semilogx(xi_ell['s'], xi_ell['correlation_2'], label='quadrupole')
    plt.semilogx(xi_ell['s'], xi_ell['correlation_4'], label='hexadecapole')

    plt.xlabel(r"$r$")
    plt.ylabel(r"$Q_\ell(r)$")
    plt.legend()

    plt.savefig(f"{PATHOUT}window_corr_{{:.2f}}sky.pdf".format(SKY_FRAC))
