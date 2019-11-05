"""View sampled likelihood values.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz


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

    likelihoods = np.exp(likelihoods)
    avg_likelihood = np.exp(avg_likelihood)

    distribution = np.array(
        [cumtrapz(values, parameters, initial=0.) for values in likelihoods]
    )
    avg_distribution = cumtrapz(avg_likelihood, parameters, initial=0.)

    if scaling == 'log':
        avg_likelihood = np.log(avg_likelihood / avg_distribution[-1])
        likelihoods = np.log(likelihoods / distribution[:, [-1]])
    elif scaling == 'normalised':
        avg_likelihood /= avg_distribution[-1]
        likelihoods /= distribution[:, [-1]]

    sns.set(style='ticks', font='serif')

    fig = plt.figure()

    plt.plot(parameters, avg_likelihood, lw=2., label="average")

    if estimate == 'max':
        max_likelihood_parameter = parameters[np.argmax(avg_likelihood)]
        lower_bound = parameters[np.argmin(
            np.abs(avg_distribution - 0.317/2*avg_distribution[-1])
        )]
        upper_bound = parameters[np.argmin(
            np.abs(avg_distribution - (1-0.317/2)*avg_distribution[-1])
        )]
        plt.axvline(
            x=max_likelihood_parameter,
            ls='--',
            label="max. loc. {}".format(max_likelihood_parameter)
        )
        plt.axvline(x=lower_bound, ls='--')
        plt.axvline(x=upper_bound, ls='--')
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
