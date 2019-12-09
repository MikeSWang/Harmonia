"""View sampled likelihood values.

"""
from __future__ import division

from collections import Iterable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import cumtrapz, simps

ONE_SIGMA_QUANTILES = [0.158655, 0.841345]
SIGMA_LEVELS = [0.954500, 0.682689, 0.000001]


def view_pdf(samples, xlabel, ylabel, scaling='normalised', estimate='max',
             truth=None, plot_range=(), precision=None, scatter_plot=True,
             xlim=None, ylim=None, fig=None):
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
    estimate : {'max', 'median', None}, optional
        If ``'max'`` (default), the maximum likelihood parameter estimate
        is marked; if ``'median'``, the posterior median; or if `None`, not
        marked.
    truth : float or None, optional
        If not `None` (default), the true parameter value is marked.
    plot_range : float, array_like, optional
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

    if plot_range:
        if len(plot_range) != 2:
            raise ValueError(
                "Normalisation range must be given as exactly two end points. "
            )
        included_range = slice(
            np.argmin(np.abs(samples['parameter'] - plot_range[0])),
            np.argmin(np.abs(samples['parameter'] - plot_range[1])) + 1
        )
    else:
        included_range = slice(None, None)

    parameters = samples['parameter'][included_range]
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

    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    main_line = plt.plot(parameters, avg_likelihood, lw=2.)

    if estimate:
        if estimate == 'max':
            estimate = parameters[np.argmax(avg_likelihood)]
        elif estimate == 'median':
            estimate = parameters[np.argmin(np.abs(
                avg_distribution - 0.5*avg_distribution[-1]
            ))]
        lower_bound = parameters[np.argmin(np.abs(
            avg_distribution - ONE_SIGMA_QUANTILES[0]*avg_distribution[-1]
        ))]
        upper_bound = parameters[np.argmin(np.abs(
            avg_distribution - ONE_SIGMA_QUANTILES[-1]*avg_distribution[-1]
        ))]
        if precision is not None:
            estimate = np.around(estimate, decimals=precision)
            lower_uncertainty = np.around(
                estimate - lower_bound, decimals=precision
            )
            upper_uncertainty = np.around(
                upper_bound - estimate, decimals=precision
            )
            if precision == 0:
                estimate = int(estimate)
                lower_uncertainty = int(lower_uncertainty)
                upper_uncertainty = int(upper_uncertainty)
        plt.axvline(
            estimate,
            ls='--', c=main_line[-1].get_color(),
            label=r"estimate ${{{}}}^{{+{}}}_{{-{}}}$".format(
                estimate, upper_uncertainty, lower_uncertainty
            )
        )
        plt.axvline(x=lower_bound, ls='--', c=main_line[-1].get_color())
        plt.axvline(x=upper_bound, ls='--', c=main_line[-1].get_color())

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


def view_contour(samples, xlabel, ylabel, truth=None, estimate=True,
                 precision=None, plot_ranges=None, cmap=None, fig=None):
    """View sampled likelihood contours.

    Parameters
    ----------
    samples : dict
        Sampled likelihood-related values.
    xlabel, ylabel : str
        Horizontal or vertical parameter label.
    truth : tuple of float or None, optional
        If not `None` (default), the true parameter values are marked.
    estimate : bool, optional
        If `True` (default), the parameter estimates and uncertainties are
        marked.
    precision : int ot None, optional
        Number of decimal places to display in estimate labels.
    plot_ranges : tuple of tuple of (float, float) or None, optional
        Horizontal or vertical axis limits.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    xx, yy = np.meshgrid(
        samples['parameter_x'], samples['parameter_y'], indexing='ij'
    )
    hh = np.exp(np.average(samples['likelihood'], axis=0)).T

    mass = simps(
        [
            simps(likelihood_y, samples['parameter_y'])
            for likelihood_y in hh
        ],
        samples['parameter_x']
    )
    hh /= mass

    h_flat = np.flip(np.sort(hh.flatten()))
    cum_h = np.cumsum(h_flat)
    cum_h /= cum_h[-1]

    h_levels = np.zeros_like(SIGMA_LEVELS)
    for n, quantile in enumerate(SIGMA_LEVELS):
        try:
            h_levels[n] = h_flat[cum_h <= quantile][-1]
        except IndexError:
            h_levels[n] = h_flat[0]

    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    else:
        plt.figure(fig.number)

    if cmap is None:
        cmap = ListedColormap(sns.color_palette('BuGn_d'))

    main = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)

    plt.contourf(
        xx, yy, hh, h_levels,
        antialiased=True, cmap=cmap
    )

    if plot_ranges:
        try:
            plt.xlim(*plot_ranges[0])
        except TypeError:
            pass
        try:
            plt.ylim(*plot_ranges[1])
        except TypeError:
            pass

    if truth:
        try:
            plt.axvline(truth[0], c='k', ls=':', lw=1.)
        except (TypeError, IndexError):
            pass
        try:
            plt.axhline(truth[1], c='k', ls=':', lw=1.)
        except (TypeError, IndexError):
            pass

    ax_ul = plt.subplot2grid((4, 4), (0, 0), rowspan=1, colspan=3, sharex=main)

    marginal_x = np.array(
        [simps(likelihood_y, samples['parameter_y']) for likelihood_y in hh]
    )

    plt.plot(samples['parameter_x'], marginal_x, c=cmap(cmap.N))

    cumulative_x = cumtrapz(marginal_x, samples['parameter_x'], initial=0.)

    if truth:
        try:
            plt.axvline(truth[0], c='k', ls=':', lw=1.)
        except (TypeError, IndexError):
            pass

    if estimate:
        median_idx = np.argmin(
            np.abs(cumulative_x - 0.5*cumulative_x[-1])
        )
        median_estimate_x = samples['parameter_x'][median_idx]

        lower_bound_idx = np.argmin(
            np.abs(cumulative_x - ONE_SIGMA_QUANTILES[0]*cumulative_x[-1])
        )
        lower_bound_x = samples['parameter_x'][lower_bound_idx]

        upper_bound_idx = np.argmin(
            np.abs(cumulative_x - ONE_SIGMA_QUANTILES[-1]*cumulative_x[-1])
        )
        upper_bound_x = samples['parameter_x'][upper_bound_idx]

        plt.vlines(
            median_estimate_x,
            ymin=0., ymax=marginal_x[median_idx],
            colors=cmap(cmap.N), linestyles='--', linewidth=1.
        )
        plt.fill_between(
            samples['parameter_x'][lower_bound_idx:upper_bound_idx+1],
            marginal_x[lower_bound_idx:upper_bound_idx+1],
            color=cmap(cmap.N), alpha=0.33, antialiased=True
        )

    ax_ul.spines['top'].set_visible(False)
    ax_ul.spines['left'].set_visible(False)
    ax_ul.spines['right'].set_visible(False)
    plt.tick_params(
        axis='x', which='both', direction='in',
        top=False, labelbottom=False
    )
    plt.tick_params(
        axis='y', which='both',
        left=False, right=False, labelleft=False
    )

    if plot_ranges:
        try:
            plt.xlim(*plot_ranges[0])
        except SyntaxError:
            pass
    plt.ylim(bottom=0, top=1.01*max(marginal_x))

    ax_lr = plt.subplot2grid((4, 4), (1, 3), rowspan=3, colspan=1, sharey=main)

    marginal_y = [
        simps(likelihood_x, samples['parameter_x'])
        for likelihood_x in hh.T
    ]

    plt.plot(marginal_y, samples['parameter_y'], c=cmap(cmap.N))

    cumulative_y = cumtrapz(marginal_y, samples['parameter_y'], initial=0.)

    if truth:
        try:
            plt.axvline(truth[1], c='k', ls=':', lw=1.)
        except (TypeError, IndexError):
            pass

    if estimate:
        median_idx = np.argmin(
            np.abs(cumulative_y - 0.5*cumulative_y[-1])
        )
        median_estimate_y = samples['parameter_y'][median_idx]

        lower_bound_idx = np.argmin(
            np.abs(cumulative_y - ONE_SIGMA_QUANTILES[0]*cumulative_y[-1])
        )
        lower_bound_y = samples['parameter_y'][lower_bound_idx]

        upper_bound_idx = np.argmin(
            np.abs(cumulative_y - ONE_SIGMA_QUANTILES[-1]*cumulative_y[-1])
        )
        upper_bound_y = samples['parameter_y'][upper_bound_idx]

        plt.hlines(
            median_estimate_y,
            xmin=0., xmax=marginal_y[median_idx],
            colors=cmap(cmap.N), linestyles='--', linewidth=1.
        )
        plt.fill_betweenx(
            samples['parameter_y'][lower_bound_idx:upper_bound_idx+1],
            marginal_y[lower_bound_idx:upper_bound_idx+1],
            edgecolor='none',
            color=cmap(cmap.N),
            alpha=0.33,
            antialiased=True
        )

    ax_lr.spines['top'].set_visible(False)
    ax_lr.spines['bottom'].set_visible(False)
    ax_lr.spines['right'].set_visible(False)
    plt.tick_params(
        axis='x', which='both',
        bottom=False, top=False, labelbottom=False
    )
    plt.tick_params(
        axis='y', which='both', direction='in',
        right=False, labelleft=False
    )

    if plot_ranges:
        try:
            plt.ylim(*plot_ranges[1])
        except SyntaxError:
            pass
    plt.xlim(left=0, right=1.01*max(marginal_y))

    if estimate:
        max_indices = np.unravel_index(np.argmax(hh), hh.shape)
        main.scatter(
            xx[max_indices], yy[max_indices],
            s=64, c=cmap(cmap.N), marker='+', label='maximum'
        )
        main.scatter(
            median_estimate_x, median_estimate_y,
            s=64, c=cmap(cmap.N), marker='x', label='median'
        )

    main.legend(handletextpad=0)

    if estimate:
        if isinstance(precision, Iterable):
            median_estimate_x = np.around(median_estimate_x, decimals=precision[0])
            lower_uncertainty_x = np.around(
                median_estimate_x - lower_bound_x, decimals=precision[0]
            )
            upper_uncertainty_x = np.around(
                upper_bound_x - median_estimate_x, decimals=precision[0]
            )
            if precision[0] == 0:
                median_estimate_x = int(median_estimate_x)
                lower_uncertainty_x = int(lower_uncertainty_x)
                upper_uncertainty_x = int(upper_uncertainty_x)

            median_estimate_y = np.around(median_estimate_y, decimals=precision[1])
            lower_uncertainty_y = np.around(
                median_estimate_y - lower_bound_y, decimals=precision[1]
            )
            upper_uncertainty_y = np.around(
                upper_bound_y - median_estimate_y, decimals=precision[1]
            )
            if precision[1] == 0:
                median_estimate_y = int(median_estimate_y)
                lower_uncertainty_y = int(lower_uncertainty_y)
                upper_uncertainty_y = int(upper_uncertainty_y)
        main.set_xlabel(
            r"{} = ${{{}}}^{{+{}}}_{{-{}}}$".format(
                xlabel, median_estimate_x, upper_uncertainty_x, lower_uncertainty_x
            )
        )
        main.set_ylabel(
            r"{} = ${{{}}}^{{+{}}}_{{-{}}}$".format(
                ylabel, median_estimate_y, upper_uncertainty_y, lower_uncertainty_y
            )
        )
    else:
        main.set_xlabel(xlabel)
        main.set_ylabel(ylabel)

    return fig
