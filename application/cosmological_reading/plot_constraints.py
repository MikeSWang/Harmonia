"""Plot posterior evaluations across different datasets.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz, simps

LINE_SCATTER_ALPHA = 1./10.
AREA_FILL_ALPHA = 1./2.
CONTOUR_FILL_ALPHA = 1./25.
ONE_SIGMA_QUANTILES = [0.158655, 0.841345]
SIGMA_LEVELS = [0.864665, 0.393469, 0.000001]


def plot_1d_constraints(posteriors, x, fig=None, label='', colour=None,
                        x_label=None, x_range=None, aggregation='average',
                        estimation=None, precision=None, scatter_plot=False):
    """Plot 1-d constraints.

    Parameters
    ----------
    posteriors : float, array_like
        (Sequence of) logarithmic posterior evaluations.
    x : float, array_like
        Parameter coordinates.
    fig : :class:`matplotlib.figure.Figure` *or None, optional*
        Existing figure object to plot on (default is `None`).
    label : str, optional
        Constraint label (default is '').
    colour : str or None, optional
        Principal colour (default is `None`).
    x_label : str or None
        Parameter label (default is `None`).
    x_range : sequence or None
        Parameter range as a sequence of length 2 (default is `None`).
    aggregation : {'average', 'combine'}, optional
        How to aggregate posterior evaluations, either by averaging
        or combining (default is 'average').
    estimation : {'median', 'maximum', None}, optional
        Parameter estimate type, if any (default is `None`).
    precision : int or None, optional
        Parameter precision as a number of decimal places (default is
        `None`).
    scatter_plot : bool, optional
        If `True`, Plot individual posteriors evaluations (default is
        `False`).

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Plotted figure object.
    estimate : tuple of float
        Parameter estimate with low and upper uncertainties.

    """
    # Set up the plottable grid.
    if x_range:
        x_selector = slice(
            np.argmin(np.abs(x - x_range[0])),
            np.argmin(np.abs(x - x_range[1])) + 1
        )
    else:
        x_selector = slice(None)

    x, posts = np.asarray(x)[x_selector], np.asarray(posteriors)[:, x_selector]

    # Aggregate posteriors and safely exponentiate.
    if aggregation == 'average':
        posterior = np.average(posts, axis=0)
    elif aggregation == 'combine':
        posterior = np.sum(posts, axis=0)
    else:
        raise ValueError(f"Invalid aggregation: {aggregation}.")

    posts = np.exp(posts - np.max(posts, axis=-1, keepdims=True))
    posterior = np.exp(posterior - np.max(posterior))

    # Normalisation with CDF.
    cdfs = np.asarray([cumtrapz(pdf, x, initial=0.) for pdf in posts])
    cum_dist_function = cumtrapz(posterior, x, initial=0.)

    posts /= cdfs[:, [-1]]
    posterior /= cum_dist_function[-1]

    cdfs /= cdfs[:, [-1]]
    cum_dist_function /= cum_dist_function[-1]

    # Plot to specifications.
    if fig is None:
        fig, canvas = plt.subplots()
    else:
        canvas, *_ = fig.axes

    posterior_plot = canvas.plot(x, posterior, c=colour, zorder=3, label=label)
    posterior_colour = posterior_plot[-1].get_color()

    if scatter_plot:
        for pdf in posts:
            canvas.plot(
                x, pdf, c=posterior_colour, zorder=1, alpha=LINE_SCATTER_ALPHA
            )

    if estimation:
        # Obtain estimates and uncertainties.
        if estimation == 'maximum':
            x_fit_idx = np.argmax(posterior)
        elif estimation == 'median':
            x_fit_idx = np.argmin(np.abs(cum_dist_function - 1./2.))

        x_lower_idx = np.argmin(
            np.abs(cum_dist_function - ONE_SIGMA_QUANTILES[0])
        )
        x_upper_idx = np.argmin(
            np.abs(cum_dist_function - ONE_SIGMA_QUANTILES[-1])
        )

        x_fit, x_lower, x_upper = x[x_fit_idx], x[x_lower_idx], x[x_upper_idx]

        dx_lower, dx_upper = x_fit - x_lower, x_upper - x_fit

        estimate = x_fit, dx_lower, dx_upper

        # Trim estimates and uncertainties.
        if precision is None:
            x_fit_disp, dx_lower_disp, dx_upper_disp = \
                x_fit, dx_lower, dx_upper
        elif precision == 0:
            x_fit_disp, dx_lower_disp, dx_upper_disp = \
                map(round, (x_fit, dx_lower, dx_upper))
        else:
            x_fit_disp = np.around(x_fit, decimals=precision)
            dx_lower_disp = np.around(dx_lower, decimals=precision)
            dx_upper_disp = np.around(dx_upper, decimals=precision)

        # Mark estimates and uncertainties.
        canvas.vlines(
            x_fit, ymin=0., ymax=posterior[x_fit_idx],
            color=posterior_colour, ls='--', zorder=3,
            label=r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
                x_label + '=' if x_label else '',
                x_fit_disp, dx_lower_disp, dx_upper_disp
            )
        )
        canvas.fill_between(
            x[x_lower_idx:(x_upper_idx + 1)],
            posterior[x_lower_idx:(x_upper_idx + 1)],
            antialiased=True,
            facecolor=posterior_colour, edgecolor='none',
            alpha=AREA_FILL_ALPHA, label=label, zorder=2
        )

    return fig, estimate


def plot_2d_constraints(posteriors, x, y, fig=None, cmap=None, alpha=None,
                        x_label=None, y_label=None, x_range=None, y_range=None,
                        aggregation='average', estimation=None,
                        x_precision=None, y_precision=None,
                        scatter_plot=False, line_style='-'):
    """Plot 2-d constraints.

    Parameters
    ----------
    posteriors : float, array_like
        (Sequence of) logarithmic posterior evaluations.
    x, y : float, array_like
        Parameter coordinates.
    fig : :class:`matplotlib.figure.Figure` *or None*, optional
        Existing figure object to plot on (default is `None`).
    cmap : str or None, optional
        Principal colour map (default is `None`).
    alpha : str or None, optional
        Principal alpha transparency (default is `None`).
    x_label, y_label : str or None
        Parameter label (default is `None`).
    x_range, y_range : sequence or None
        Parameter range as a sequence of length 2 (default is `None`).
    aggregation : {'average', 'combine'}, optional
        How to aggregate posterior evaluations, either by averaging
        or combining (default is 'average').
    estimation : {'median', 'maximum', None}, optional
        Parameter estimate type, if any (default is `None`).
    x_precision, y_precision : int or None, optional
        Parameter precision as a number of decimal places (default is
        `None`).
    scatter_plot : bool, optional
        If `True`, plot individual posteriors evaluations (default is
        `False`).
    line_style : str, optional
        Line style for the contours (default is '-').

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Plotted figure object.
    x_estimate, y_estimate : tuple of float
        Parameter estimates with low and upper uncertainties.
    patch : :class:`matplotlib.patches.Rectangle`
        A colour patch to be used in the legend.

    """
    # Set up the plottable grid.
    if x_range:
        x_selector = slice(
            np.argmin(np.abs(x - x_range[0])),
            np.argmin(np.abs(x - x_range[1])) + 1
        )
    else:
        x_selector = slice(None)

    if y_range:
        y_selector = slice(
            np.argmin(np.abs(y - y_range[0])),
            np.argmin(np.abs(y - y_range[1])) + 1
        )
    else:
        y_selector = slice(None)

    x, y = np.asarray(x)[x_selector], np.asarray(y)[y_selector]
    xx, yy = np.meshgrid(x, y, indexing='ij')
    posts = np.asarray(posteriors)[:, x_selector, y_selector]

    # Aggregate posteriors and safely exponentiate.
    if aggregation == 'average':
        posterior = np.average(posts, axis=0)
    elif aggregation == 'combine':
        posterior = np.sum(posts, axis=0)
    else:
        raise ValueError(f"Invalid aggregation: {aggregation}.")

    posts = np.exp(posts - np.max(posts, axis=(-2, -1), keepdims=True))
    posterior = np.exp(posterior - np.max(posterior))

    # Normalisation with probability mass.
    def _prob_mass(hh):
        return simps([simps(hh_xslice, y) for hh_xslice in hh], x)

    posts = [post / _prob_mass(post) for post in posts]
    posterior /= _prob_mass(posterior)

    # Plot to specifications.
    if fig is None:
        fig = plt.figure()
        main_panel = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
        x_panel = plt.subplot2grid(
            (4, 4), (0, 0), colspan=3, sharex=main_panel
        )
        y_panel = plt.subplot2grid(
            (4, 4), (1, 3), rowspan=3, sharey=main_panel
        )
    else:
        main_panel, x_panel, y_panel = fig.axes

    # Likelihood contours.
    def _plot_contours(hh, relative_alpha, zorder=None):

        # Get posterior surface heights on grid and cumulative heights.
        h_flat = np.flip(np.sort(hh.flatten()))
        cum_h = np.cumsum(h_flat)
        cum_h /= cum_h[-1]

        # Determine posterior surface height levels.
        h_levels = np.zeros_like(SIGMA_LEVELS)
        for n_sigma, sigma_level in enumerate(SIGMA_LEVELS):
            try:
                h_levels[n_sigma] = h_flat[cum_h <= sigma_level][-1]
            except IndexError:
                h_levels[n_sigma] = h_flat[0]

        # Plot the contour.
        try:
            contour = main_panel.contourf(
                xx, yy, hh, h_levels, cmap=cmap, antialiased=True,
                alpha=relative_alpha*alpha, zorder=zorder or 1
            )
            main_panel.contour(
                contour,
                linestyles=line_style, colors=contour.cmap(contour.cmap.N),
                alpha=min(2*relative_alpha*alpha, 1.), zorder=zorder or 1
            )
        except ValueError as error:
            if str(error) == "Contour levels must be increasing":
                raise ValueError(
                    "Cannot process posterior values into contours."
                ) from error
            raise ValueError from error

        return contour

    main_contour = _plot_contours(posterior, relative_alpha=1., zorder=2)
    patch = plt.Rectangle(
        (0., 0.), 2., 1., ls=line_style,
        ec=None, fc=main_contour.cmap(main_contour.cmap.N), alpha=0.8,
        # main_contour.collections[-1].get_facecolor()[0]
    )
    if scatter_plot:
        scattered_contours = []
        for post in posts:
            individual_contour = _plot_contours(
                post, relative_alpha=CONTOUR_FILL_ALPHA
            )
            scattered_contours.append(individual_contour)

    # Likelihood marginal PDFs.
    def _plot_pdfs(hh, relative_alpha, cm, estimate=False, zorder=None):

        pdf_x = np.asarray([simps(hh_xslice, y) for hh_xslice in hh])
        pdf_y = np.asarray([simps(hh_yslice, x) for hh_yslice in hh.T])
        cdf_x = cumtrapz(pdf_x, x, initial=0.)
        cdf_y = cumtrapz(pdf_y, y, initial=0.)

        pdf_x /= cdf_x[-1]
        pdf_y /= cdf_y[-1]
        cdf_x /= cdf_x[-1]
        cdf_y /= cdf_y[-1]

        x_panel.plot(
            x, pdf_x, c=cm(cm.N), ls=line_style,
            alpha=relative_alpha, zorder=zorder or 1
        )
        y_panel.plot(
            pdf_y, y, c=cm(cm.N), ls=line_style,
            alpha=relative_alpha, zorder=zorder or 1
        )

        if estimate:
            if estimation == 'maximum':
                x_fit_idx = np.argmax(pdf_x)
                y_fit_idx = np.argmax(pdf_y)
            elif estimation == 'median':
                x_fit_idx = np.argmin(np.abs(cdf_x - 1./2.))
                y_fit_idx = np.argmin(np.abs(cdf_y - 1./2.))

            x_lower_idx = np.argmin(np.abs(cdf_x - ONE_SIGMA_QUANTILES[0]))
            y_lower_idx = np.argmin(np.abs(cdf_y - ONE_SIGMA_QUANTILES[0]))
            x_upper_idx = np.argmin(np.abs(cdf_x - ONE_SIGMA_QUANTILES[-1]))
            y_upper_idx = np.argmin(np.abs(cdf_y - ONE_SIGMA_QUANTILES[-1]))

            x_fit, x_lower, x_upper = x[[x_fit_idx, x_lower_idx, x_upper_idx]]
            y_fit, y_lower, y_upper = y[[y_fit_idx, y_lower_idx, y_upper_idx]]

            dx_lower, dx_upper = x_fit - x_lower, x_upper - x_fit
            dy_lower, dy_upper = y_fit - y_lower, y_upper - y_fit

            if x_precision is None:
                x_fit_disp, dx_lower_disp, dx_upper_disp = \
                    x_fit, dx_lower, dx_upper
            elif x_precision == 0:
                x_fit_disp, dx_lower_disp, dx_upper_disp = \
                    map(int, map(round, (x_fit, dx_lower, dx_upper)))
            else:
                x_fit_disp = np.around(x_fit, decimals=x_precision)
                dx_lower_disp = np.around(dx_lower, decimals=x_precision)
                dx_upper_disp = np.around(dx_upper, decimals=x_precision)

            if y_precision is None:
                y_fit_disp, dy_lower_disp, dy_upper_disp = \
                    y_fit, dy_lower, dy_upper
            elif y_precision == 0:
                y_fit_disp, dy_lower_disp, dy_upper_disp = \
                    map(int, map(round, (y_fit, dy_lower, dy_upper)))
            else:
                y_fit_disp = np.around(y_fit, decimals=y_precision)
                dy_lower_disp = np.around(dy_lower, decimals=y_precision)
                dy_upper_disp = np.around(dy_upper, decimals=y_precision)

            x_panel.vlines(
                x_fit, ymin=0., ymax=pdf_x[x_fit_idx],
                linestyles='--', colors=[cm(cm.N)], zorder=3,
                label=r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
                    x_label + '=' if x_label else '',
                    x_fit_disp, dx_lower_disp, dx_upper_disp
                )
            )
            x_panel.fill_between(
                x[x_lower_idx:(x_upper_idx + 1)],
                pdf_x[x_lower_idx:(x_upper_idx + 1)],
                antialiased=True, facecolor=[cm(cm.N)], edgecolor='none',
                alpha=AREA_FILL_ALPHA, zorder=2
            )

            y_panel.hlines(
                y_fit, xmin=0., xmax=pdf_y[y_fit_idx],
                linestyles='--', colors=[cm(cm.N)], zorder=3,
                label=r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
                    y_label + '=' if y_label else '',
                    y_fit_disp, dy_lower_disp, dy_upper_disp
                )
            )
            y_panel.fill_betweenx(
                y[y_lower_idx:(y_upper_idx + 1)],
                pdf_y[y_lower_idx:(y_upper_idx + 1)],
                antialiased=True, facecolor=[cm(cm.N)], edgecolor='none',
                alpha=AREA_FILL_ALPHA, zorder=2
            )

            return (x_fit, x_lower, x_upper), (y_fit, y_lower, y_upper)

        return None, None

    x_result, y_result = _plot_pdfs(
        posterior, estimate=True, relative_alpha=1.,
        cm=main_contour.cmap, zorder=3
    )
    if scatter_plot:
        for post, individual_contour in zip(posts, scattered_contours):
            _plot_pdfs(
                post,
                relative_alpha=LINE_SCATTER_ALPHA, cm=individual_contour.cmap
            )

    if estimation == 'median':
        main_panel.scatter(
            x_result[0], y_result[0], marker='+', s=40,
            c=[main_contour.cmap(main_contour.cmap.N)], zorder=3
        )
    elif estimation == 'maximum':
        # Note this provides the joint maximum posterior estimates
        # not the marginal maximum posterior estimates.
        xy_fit_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
        main_panel.scatter(
            xx[xy_fit_idx], yy[xy_fit_idx], marker='+', s=40,
            c=[main_contour.cmap(main_contour.cmap.N)],
            zorder=3
        )

    return fig, x_result, y_result, patch
