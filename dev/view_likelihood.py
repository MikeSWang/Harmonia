"""View spherical likelihoods.

"""
import seaborn as sns
from matplotlib import pyplot as plt


def view_chi_square(data):
    """View data--model chi-square values.

    Parameters
    ----------
    data : dict
        Chi-square samples.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    sns.set(style='ticks', font='serif')

    parameters = data['f_nl']
    chi_square = data['chi_square']

    plt.figure("Non-Gaussianity chi-square")
    plt.plot(parameters, chi_square)
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
