"""View sampled likelihood values.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz

ONE_SIGMA_QUANTILES = [0.158655, 0.841345]


def view_samples(samples, xlabel, ylabel, scaling='normalised', estimate='max',
                 truth=None, norm_range=(), precision=None, scatter_plot=True,
                 xlim=None, ylim=None):
    """View sampled likelihood-related values.

    Parameters
    ----------
    samples : dict
        Sampled likelihood-related values.
    xlabel, ylabel : str
        Horizontal or vertical axis label.
    scaling : {'log', 'normalised'}, optional
        If ``'log'``, plot the logarithmic likelihood; if ``'normalised'``
        (default), plot the likelihood normalised to full integral value of
        unity.
    estimate : {'max', None}, optional
        If ``'max'`` (default), the likelihood parameter estimate is marked
        or if `None`, not marked.
    truth : float or None, optional
        If not `None` (default), the true parameter value is marked.
    norm_range : float, array_like, optional
        If not an empty tuple (default), normalise likelihood in the
        parameter range specified.
    precision : int ot None, optional
        Number of decimal places to display in estimate labels.
    scatter_plot : bool, optional
        If `True` (default), each realisation in the sampling data is
        plotted in additional to the overall sample averages.
    xlim, ylim : tuple of (float, float), optional
        Horizontal or vertical axis limits.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    SCATTER_ALPHA = 1/8

    if norm_range:
        if len(norm_range) != 2:
            raise ValueError(
                "Normalisation range must be given as exactly two end points. "
            )
        # Firs [0] for returning the first and only axis, and the second
        # [0] for the first and only index.
        try:
            included_range = slice(
                np.argwhere(samples['parameters'] == norm_range[0])[0][0],
                np.argwhere(samples['parameters'] == norm_range[1])[0][0] + 1
            )
        except IndexError:
            included_range = slice(None, None)
    else:
        included_range = slice(None, None)

    parameters = samples['parameters'][included_range]
    likelihoods = samples['likelihood'][:, included_range]

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

    plt.plot(parameters, avg_likelihood, lw=2.)

    if estimate == 'max':
        max_likelihood_parameter = parameters[np.argmax(avg_likelihood)]
        lower_bound = parameters[np.argmin(np.abs(
            avg_distribution - ONE_SIGMA_QUANTILES[0]*avg_distribution[-1]
        ))]
        upper_bound = parameters[np.argmin(np.abs(
            avg_distribution - ONE_SIGMA_QUANTILES[-1]*avg_distribution[-1]
        ))]
        if precision is not None:
            max_likelihood_parameter = np.around(
                max_likelihood_parameter, decimals=precision
            )
            lower_uncertainty = np.around(
                max_likelihood_parameter - lower_bound, decimals=precision
            )
            upper_uncertainty = np.around(
                upper_bound - max_likelihood_parameter, decimals=precision
            )
            if precision == 0:
                max_likelihood_parameter = int(max_likelihood_parameter)
                lower_uncertainty = int(lower_uncertainty)
                upper_uncertainty = int(upper_uncertainty)
        plt.axvline(
            x=max_likelihood_parameter,
            ls='--',
            label=r"estimate ${{{}}}^{{+{}}}_{{-{}}}$".format(
                max_likelihood_parameter,
                upper_uncertainty,
                lower_uncertainty
            )
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
