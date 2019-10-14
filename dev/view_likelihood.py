"""View spherical likelihoods.

"""
import seaborn as sns
from matplotlib import pyplot as plt


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

    # HACK
    SEL = slice(1, None)

    parameters = data['coords'][SEL]
    if scaling == 'normal':
        raise NotImplementedError
    elif scaling == 'log':
        likelihoods = data['logp'][SEL]

    plt.figure("{} non-Gaussianity likelihood".format(scaling))
    plt.plot(parameters, likelihoods, label="sharpened likelihood")
    plt.xlabel(r"$f_\mathrm{NL}$")
    plt.ylabel(r"$\log\mathcal{L}(f_\mathrm{NL})$")

