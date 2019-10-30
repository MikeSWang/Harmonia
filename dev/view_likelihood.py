"""View sampled likelihood values.

"""
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import simps


def view_samples(samples, xlabel, ylabel, scatter_plot=False,
                 xlim=None, ylim=None, scaling=None, estimate=None,
                 truth=None):
    """View sampled likelihood-related values.

    Parameters
    ----------
    samples : dict
        Sampled likelihood-related value.
    xlabel, ylabel : str
        Horizontal or vertical axis label.
    scatter_plot : bool, optional
        If `True` (default is `False`), each realisation in the sampling
        data is plotted in additional to the overall sample averages.
    xlim, ylim : tuple of (float, float), optional
        Horizontal or vertical axis limits.
    scaling : {'normalised', 'exp_max', None}, optional
        If `None` (default), the logarithmic likelihood is plotted;
        if ``'normalised'``, the normalised posterior with uniform prior is
        plotted; if ``'exp_max'``, the likelihood normalised to the maximum
        is plotted.
    estimate : {'max', None}, optional
        If not `None` (default), the likelihood parameter estimate is
        marked.
    truth : float or None, optional
        If not `None` (default), the true parameter value is marked.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    SCATTER_ALPHA = 1/8

    parameters = samples['parameters']
    likelihoods = samples['likelihood']

    likelihoods -= np.min(likelihoods, axis=1)[:, None]

    avg_likelihood = np.average(likelihoods, axis=0)
    avg_likelihood -= np.min(avg_likelihood)

    if scaling == 'exp_max':
        avg_likelihood = np.exp(avg_likelihood - np.max(avg_likelihood))
        likelihoods = np.exp(likelihoods - np.max(avg_likelihood, axis=1))
    elif scaling == 'normalised':
        avg_likelihood = np.exp(avg_likelihood)
        likelihoods = np.exp(likelihoods)
        avg_normalisation = simps(avg_likelihood, parameters)
        normalisations = np.array(
            [simps(values, parameters) for values in likelihoods]
        )
        avg_likelihood /= avg_normalisation
        likelihoods /= normalisations[:, None]

    sns.set(style='ticks', font='serif')

    fig = plt.figure()

    plt.plot(parameters, avg_likelihood, lw=2., label="average")

    if estimate == 'max':
        max_likelihood_parameter = parameters[np.argmax(avg_likelihood)]
        plt.axvline(
            x=max_likelihood_parameter,
            ls='--',
            label="max. loc. {}".format(max_likelihood_parameter)
        )
    if truth is not None:
        plt.axvline(x=truth, ls=':', label="truth {}".format(truth))

    if scatter_plot:
        for values in likelihoods:
            plt.plot(parameters, values, alpha=SCATTER_ALPHA)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(parameters.min(), parameters.max())
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    return fig
