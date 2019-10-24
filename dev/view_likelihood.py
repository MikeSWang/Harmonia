"""View spherical likelihoods.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def view_chi_square(data, scatter=False):
    """View data--model chi-square values.

    Parameters
    ----------
    data : dict
        Chi-square samples.
    scatter : bool, optional
        If `True` (default is `False`), data are plural and each is plotted
        scattered around the averaged data plot.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    sns.set(style='ticks', font='serif')

    plt.figure("Non-Gaussianity chi-square")

    if not scatter:
        parameters = data['f_nl']
        chi_square = data['chi_square']
        plt.plot(parameters, chi_square)
    else:
        for parameters, chi_square in zip(data['f_nl'], data['chi_square']):
            plt.plot(parameters, chi_square, alpha=1/5)
        plt.plot(
            np.average(data['f_nl'], axis=0),
            np.average(data['chi_square'], axis=0),
            lw=2.,
            ls='--'
        )

    plt.xlabel(r"$f_\mathrm{NL}$")
    plt.ylabel(r"$\chi^2(f_\mathrm{NL})$")


def view_likelihood(data, scaling='log'):
    """View likelihood function.

    Parameters
    ----------
    data : dict
        Likelihood samples.
    scaling : {'normalised', 'log'}, optional
        Plot 'normalised' or 'log' (default) likelihood function.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    sns.set(style='ticks', font='serif')

    parameters = data['f_nl']
    if scaling == 'normal':
        raise NotImplementedError
    elif scaling == 'log':
        likelihoods = data['likelihood']

    plt.figure("{} non-Gaussianity likelihood".format(scaling))
    plt.plot(parameters, likelihoods)
    plt.xlabel(r"$f_\mathrm{NL}$")
    plt.ylabel(r"$\log\mathcal{L}(f_\mathrm{NL})$")
