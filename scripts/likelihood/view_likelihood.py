"""View sampled likelihood surface.

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


def view_pdf(samples, xlabel, ylabel, scaling='normalised', precision=None,
             truth=None, estimate='max', scatter_plot=True, plot_range=(),
             xlim=None, ylim=None):
    """View sampled likelihood-related values.

    Parameters
    ----------
    samples : (list of) dict
        Sampled log-likelihood values.
    xlabel, ylabel : str
        Horizontal or vertical axis label.
    scaling : {'log', 'normalised'}, optional
        If ``'normalised'`` (default), plot the likelihood normalised to
        full integral value of unity; if ``'log'``, plot the logarithmic
        likelihood.
    precision : int or None, optional
        Number of decimal places to display.
    truth : float or None, optional
        If not `None` (default), the true parameter value is marked.
    estimate : {'max', 'median', None}, optional
        If ``'max'`` (default), the maximum likelihood parameter estimate
        is marked; if ``'median'``, the posterior median; or if `None`, not
        marked.
    scatter_plot : bool, optional
        If `True` (default), each realisation in the sampling data is
        plotted in additional to the overall sample averages.
    plot_range : float, array_like, optional
        If not an empty tuple (default), normalise likelihood in the
        parameter range specified.
    xlim, ylim : tuple of (float, float), optional
        Horizontal or vertical axis limits.

    Returns
    -------
    :class:`matplotlib.Figure`

    """
    SCATTER_ALPHA = 1/8

    if not isinstance(samples, list):
        samples = [samples]

    sns.set(style='ticks', font='serif')
    fig = plt.figure()
    fig.add_subplot(111)

    def _step_plot(_samples, _fig, _estimate):

        # Likelihood renormalisation to range.
        included_range = slice(None, None)
        if plot_range:
            assert len(plot_range) == 2, \
                "Normalisation range must be given as exactly two end points. "
            included_range = slice(
                np.argmin(np.abs(samples['parameter'] - plot_range[0])),
                np.argmin(np.abs(samples['parameter'] - plot_range[1])) + 1
            )

        parameters = _samples['parameter'][included_range]
        log_likelihoods = _samples['likelihood'][:, included_range]

        log_likelihoods -= np.min(log_likelihoods, axis=1)[:, None]
        avg_log_likelihood = np.average(log_likelihoods, axis=0)

        likelihoods = np.exp(log_likelihoods)
        avg_likelihood = np.exp(avg_log_likelihood)

        distribution = np.asarray([
            cumtrapz(values, parameters, initial=0.) for values in likelihoods
        ])
        avg_distribution = cumtrapz(avg_likelihood, parameters, initial=0.)

        likelihoods /= distribution[:, [-1]]
        avg_likelihood /= avg_distribution[-1]
        if scaling == 'log':
            avg_likelihood = np.log(avg_likelihood)
            likelihoods = np.log(likelihoods)

        distribution /= distribution[:, [-1]]
        avg_distribution /= avg_distribution[-1]

        # Plo to specifications.
        main, *_ = _fig.axes

        summary_line = main.plot(parameters, avg_likelihood, lw=2.)

        if _estimate:
            if _estimate == 'max':
                estimate = parameters[np.argmax(avg_likelihood)]
            elif _estimate == 'median':
                estimate = parameters[np.argmin(np.abs(avg_distribution-0.5))]
            lower_bound = parameters[np.argmin(
                np.abs(avg_distribution - ONE_SIGMA_QUANTILES[0])
            )]
            upper_bound = parameters[np.argmin(
                np.abs(avg_distribution - ONE_SIGMA_QUANTILES[-1])
            )]
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
            main.axvline(
                estimate, ls='--', c=summary_line[-1].get_color(),
                label=r"estimate ${{{}}}^{{+{}}}_{{-{}}}$".format(
                    estimate, upper_uncertainty, lower_uncertainty
                )
            )
            main.axvline(
                x=lower_bound, ls='--',
                c=summary_line[-1].get_color()
            )
            main.axvline(
                x=upper_bound, ls='--',
                c=summary_line[-1].get_color()
            )

        if scatter_plot:
            for values in likelihoods:
                main.plot(parameters, values, alpha=SCATTER_ALPHA)

        main.set_xlim(parameters.min(), parameters.max())

        return _fig

    for samples_to_plot in samples:
        fig = _step_plot(samples_to_plot, fig, estimate)

    main, *_ = fig.axes

    if truth is not None:
        main.axvline(x=truth, ls=':', label="truth {}".format(truth))

    if xlim:
        main.set_xlim(xlim)
    if ylim:
        main.set_ylim(ylim)

    main.set_xlabel(xlabel)
    main.set_ylabel(ylabel)
    main.legend()

    return fig


def view_contour(samples, xlabel, ylabel, precision=None, truth=None,
                 estimate='max', plot_ranges=None, cmap=None, alpha=None):
    """View sampled likelihood contours.

    Parameters
    ----------
    samples : (list of) dict
        Sampled log-likelihood values.
    xlabel, ylabel : str
        Horizontal or vertical parameter label.
    precision : tuple of int or None, optional
        Number of decimal places to display.
    truth : tuple of float or None, optional
        If not `None` (default), the true parameter values are marked.
    estimate : {'max', 'median', None}, optional
        If ``'max'`` (default), the maximum likelihood parameter estimate
        is marked; if ``'median'``, the posterior median; or if `None`, not
        marked.
    plot_ranges : tuple of list of float or None, optional
        Horizontal abd vertical axis limits.
    cmap : (list of) :class:`matplotlib.ScalarMappable`
        Colour map(s) for sampled likelihood contours.
    alpha : (list of) float
        Transparency alpha(s) for sampled likelihood contours.

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.
    estimates : (list of) dict
        Estimated results.

    """
    if not isinstance(samples, list):
        samples = [samples]
    if cmap is None:
        cmap = [None] * len(samples)
    else:
        cmap = list(
            map(lambda name: ListedColormap(sns.color_palette(name)), cmap)
        )
    if alpha is None:
        alpha = list(np.linspace(1, 0.75, len(samples)))

    fig = plt.figure(figsize=(6, 6))

    canvas = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
    top_panel = plt.subplot2grid(
        (4, 4), (0, 0), rowspan=1, colspan=3, sharex=canvas
    )
    right_panel = plt.subplot2grid(
        (4, 4), (1, 3), rowspan=3, colspan=1, sharey=canvas
    )

    def _step_plot(_samples, _fig, _estimate, _cmap, _alpha):

        main, xpanel, ypanel = _fig.axes

        # Parameter and likelihood grids.
        xx, yy = np.meshgrid(
            _samples['parameter_x'], _samples['parameter_y'], indexing='ij'
        )

        log_hh = np.average(_samples['likelihood'], axis=0)
        hh = np.squeeze(np.exp(log_hh - np.min(log_hh))).T

        # Likelihood normalisation.
        mass = simps(
            [
                simps(likelihood_y, _samples['parameter_y'])
                for likelihood_y in hh
            ],
            _samples['parameter_x']
        )
        hh /= mass

        # Likelihood level sets.
        h_flat = np.flip(np.sort(hh.flatten()))
        cum_h = np.cumsum(h_flat)
        cum_h /= cum_h[-1]

        h_levels = np.zeros_like(SIGMA_LEVELS)
        for n, quantile in enumerate(SIGMA_LEVELS):
            try:
                h_levels[n] = h_flat[cum_h <= quantile][-1]
            except IndexError:
                h_levels[n] = h_flat[0]

        # Contour plot.
        if _cmap is None:
            _cmap = ListedColormap(sns.color_palette('Greens'))

        contour = main.contourf(
            xx, yy, hh, h_levels,
            antialiased=True, cmap=_cmap, alpha=_alpha
        )
        main.contour(
            contour,
            linewidths=1.25, colors=_cmap(_cmap.N), alpha=min(2*_alpha, 1.)
        )

        # Marginal likelihoods.
        marginal_x = np.asarray([
            simps(likelihood_y, _samples['parameter_y'])
            for likelihood_y in hh
        ])
        cumulative_x = cumtrapz(
            marginal_x, _samples['parameter_x'], initial=0.
        )

        marginal_x /= cumulative_x[-1]
        cumulative_x /= cumulative_x[-1]

        xpanel.plot(_samples['parameter_x'], marginal_x, c=_cmap(_cmap.N))
        if _estimate:
            if _estimate == 'max':
                estimate_idx = np.argmax(marginal_x)
            elif _estimate == 'median':
                estimate_idx = np.argmin(np.abs(cumulative_x - 0.5))

            lower_bound_idx = np.argmin(
                np.abs(cumulative_x - ONE_SIGMA_QUANTILES[0])
            )
            upper_bound_idx = np.argmin(
                np.abs(cumulative_x - ONE_SIGMA_QUANTILES[-1])
            )

            estimate_x = _samples['parameter_x'][estimate_idx]
            lower_bound_x = _samples['parameter_x'][lower_bound_idx]
            upper_bound_x = _samples['parameter_x'][upper_bound_idx]

            xpanel.vlines(
                estimate_x, ymin=0., ymax=marginal_x[estimate_idx],
                linestyles='--', linewidth=1., colors=_cmap(_cmap.N)
            )
            xpanel.fill_between(
                _samples['parameter_x'][lower_bound_idx:upper_bound_idx+1],
                marginal_x[lower_bound_idx:upper_bound_idx+1],
                color=_cmap(_cmap.N), antialiased=True, alpha=0.33
            )

        marginal_y = np.asarray([
            simps(likelihood_x, _samples['parameter_x'])
            for likelihood_x in hh.T
        ])
        cumulative_y = cumtrapz(
            marginal_y, _samples['parameter_y'], initial=0.
        )

        marginal_y /= cumulative_y[-1]
        cumulative_y /= cumulative_y[-1]

        ypanel.plot(marginal_y, _samples['parameter_y'], c=_cmap(_cmap.N))
        if _estimate:
            if _estimate == 'max':
                estimate_idx = np.argmax(marginal_y)
            elif _estimate == 'median':
                estimate_idx = np.argmin(np.abs(cumulative_y - 0.5))

            lower_bound_idx = np.argmin(
                np.abs(cumulative_y - ONE_SIGMA_QUANTILES[0])
            )
            upper_bound_idx = np.argmin(
                np.abs(cumulative_y - ONE_SIGMA_QUANTILES[-1])
            )

            estimate_y = _samples['parameter_y'][estimate_idx]
            lower_bound_y = _samples['parameter_y'][lower_bound_idx]
            upper_bound_y = _samples['parameter_y'][upper_bound_idx]

            ypanel.hlines(
                estimate_y, xmin=0., xmax=marginal_y[estimate_idx],
                linestyles='--', linewidth=1., colors=_cmap(_cmap.N)
            )
            ypanel.fill_betweenx(
                _samples['parameter_y'][lower_bound_idx:upper_bound_idx+1],
                marginal_y[lower_bound_idx:upper_bound_idx+1],
                color=_cmap(_cmap.N), antialiased=True, alpha=0.33
            )

        if _estimate:
            if isinstance(precision, Iterable):
                estimate_x = np.around(estimate_x, decimals=precision[0])
                lower_uncertainty_x = np.around(
                    estimate_x - lower_bound_x, decimals=precision[0]
                )
                upper_uncertainty_x = np.around(
                    upper_bound_x - estimate_x, decimals=precision[0]
                )
                if precision[0] == 0:
                    estimate_x = int(estimate_x)
                    lower_uncertainty_x = int(lower_uncertainty_x)
                    upper_uncertainty_x = int(upper_uncertainty_x)

                estimate_y = np.around(estimate_y, decimals=precision[1])
                lower_uncertainty_y = np.around(
                    estimate_y - lower_bound_y, decimals=precision[1]
                )
                upper_uncertainty_y = np.around(
                    upper_bound_y - estimate_y, decimals=precision[1]
                )
                if precision[1] == 0:
                    estimate_y = int(estimate_y)
                    lower_uncertainty_y = int(lower_uncertainty_y)
                    upper_uncertainty_y = int(upper_uncertainty_y)

            x_est_label = r"{} = ${{{}}}^{{+{}}}_{{-{}}}$".format(
                xlabel, estimate_x, upper_uncertainty_x, lower_uncertainty_x
            )
            y_est_label = r"{} = ${{{}}}^{{+{}}}_{{-{}}}$".format(
                ylabel, estimate_y, upper_uncertainty_y, lower_uncertainty_y
            )

            main.scatter(
                estimate_x, estimate_y, marker='+', s=64,
                c=_cmap(_cmap.N), label=", ".join([x_est_label, y_est_label])
            )

        return _fig

    for samples_to_plot, cmap_to_use, alpha_to_use in zip(samples, cmap, alpha):
        fig = _step_plot(
            samples_to_plot, fig, estimate, cmap_to_use, alpha_to_use
        )

    if truth:
        try:
            canvas.axvline(truth[0], c='k', ls=':', lw=1.)
        except (TypeError, IndexError):
            pass
        try:
            canvas.axhline(truth[1], c='k', ls=':', lw=1.)
        except (TypeError, IndexError):
            pass

    if plot_ranges:
        try:
            canvas.set_xlim(*plot_ranges[0])
        except TypeError:
            pass
        try:
            canvas.set_ylim(*plot_ranges[1])
        except TypeError:
            pass

    canvas.set_xlabel(xlabel)
    canvas.set_ylabel(ylabel)
    canvas.legend(handletextpad=0)

    top_panel.set_ylim(bottom=0)
    top_panel.axes.tick_params(
        axis='x', which='both', direction='in', top=False, labelbottom=False
    )
    top_panel.axes.tick_params(
        axis='y', which='both', left=False, right=False, labelleft=False
    )
    top_panel.spines['top'].set_visible(False)
    top_panel.spines['left'].set_visible(False)
    top_panel.spines['right'].set_visible(False)

    right_panel.set_xlim(left=0)
    right_panel.axes.tick_params(
        axis='x', which='both', top=False, bottom=False, labelbottom=False
    )
    right_panel.axes.tick_params(
        axis='y', which='both', direction='in', right=False, labelleft=False
    )
    right_panel.spines['top'].set_visible(False)
    right_panel.spines['bottom'].set_visible(False)
    right_panel.spines['right'].set_visible(False)

    return fig
