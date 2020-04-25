"""Extract cosmological constraints from sampled log-likelihood values.

"""
import glob
import os
import sys
import warnings
from collections.abc import Sequence
from itertools import tee

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import cumtrapz, simps

try:
    from application import confirm_directory, data_dir, overwrite_protection
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))
    from application import confirm_directory, data_dir, overwrite_protection

sns.set(style='ticks', font='serif')

LINE_SCATTER_ALPHA = 1./8.
CONTOUR_SCATTER_RELATIVE_ALPHA = 1./20.
AREA_FILL_ALPHA = 1./3.
ONE_SIGMA_QUANTILES = [0.158655, 0.841345]
SIGMA_LEVELS = [0.954500, 0.682689, 0.000001]


def _plot_likelihood_1d(fig, lkhds, x, x_precision, x_range, x_label,
                        label=None, estimate=None, plt_scatter=False):

    # Zoom in on grid.
    if x_range:
        x_selector = slice(
            np.argmin(np.abs(x - x_range[0])),
            np.argmin(np.abs(x - x_range[1])) + 1
        )
    else:
        x_selector = slice(None)

    x, lkhds = x[x_selector], np.asarray(lkhds)[:, x_selector]

    # Safe exponentiation.
    lkhds -= np.min(lkhds, axis=1)[:, None]
    likelihood = np.average(lkhds, axis=0)

    lkhds, likelihood = np.exp(lkhds), np.exp(likelihood)

    # Normalisation with CDF.
    cdfs = np.asarray([cumtrapz(lkhd, x, initial=0.) for lkhd in lkhds])
    cum_dist_function = cumtrapz(likelihood, x, initial=0.)

    lkhds /= cdfs[:, [-1]]
    likelihood /= cum_dist_function[-1]

    cdfs /= cdfs[:, [-1]]
    cum_dist_function /= cum_dist_function[-1]

    # Plot to specifications.
    canvas, *_ = fig.axes

    summary_lkhd_plot = canvas.plot(x, likelihood, label=label)
    summary_colour = summary_lkhd_plot[-1].get_color()

    if plt_scatter:
        for lkhd in lkhds:
            canvas.plot(x, lkhd, c=summary_colour, alpha=LINE_SCATTER_ALPHA)

    if estimate:
        # Obtain estimates and uncertainties.
        if estimate == 'maximum':
            x_fit = x[np.argmax(likelihood)]
        elif estimate == 'median':
            x_fit = x[np.argmin(np.abs(cum_dist_function - 1./2.))]

        x_lower = x[np.argmin(
            np.abs(cum_dist_function - ONE_SIGMA_QUANTILES[0])
        )]
        x_upper = x[np.argmin(
            np.abs(cum_dist_function - ONE_SIGMA_QUANTILES[-1])
        )]

        # Trim estimates and uncertainties.
        if x_precision is not None:
            x_fit = np.around(x_fit, decimals=x_precision)
            dx_lower = np.around(x_fit - x_lower, decimals=x_precision)
            dx_upper = np.around(x_upper - x_fit, decimals=x_precision)
            if x_precision == 0:
                x_fit, dx_lower, dx_upper = \
                    map(int, (x_fit, dx_lower, dx_upper))

        # Mark estimates and uncertainties.
        x_label = x_label + '=' if x_label else ''
        canvas.axvline(
            x_fit, ls=':', c=summary_colour,
            label=r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
                x_label, x_fit, dx_lower, dx_upper
            )
        )
        canvas.axvline(x=x_lower, ls=':', c=summary_colour)
        canvas.axvline(x=x_upper, ls=':', c=summary_colour)
        # canvas.annotate(
        #     r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
        #         x_label, x_fit, dx_lower, dx_upper
        #     ),
        #     xy=(0.5, 1.1),
        #     xycoords='axes fraction',
        #     verticalalignment='baseline',
        #     horizontalalignment='center'
        # )

    return fig, (x_fit, dx_lower, dx_upper)


def _plot_likelihood_2d(fig, cmap, alpha, lkhds, x, y, x_label, y_label,
                        x_precision, y_precision, x_range, y_range,
                        label=None, estimate=None, plt_scatter=False):

    main_panel, x_panel, y_panel = fig.axes

    x_label = x_label + '=' if x_label else ''
    y_label = y_label + '=' if y_label else ''

    # Zoom in on grid.
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

    x, y = x[x_selector], y[y_selector]

    xx, yy = np.meshgrid(x, y, indexing='ij')
    lkhds = np.asarray(lkhds)[:, x_selector, y_selector]

    # Safe exponentiation.
    lkhds = [lkhd - np.median(lkhd) for lkhd in lkhds]
    likelihood = np.average(lkhds, axis=0)

    lkhds, likelihood = np.exp(lkhds), np.exp(likelihood)

    # Normalisation with probability mass.
    def _prob_mass(hh):
        return simps([simps(hh_xslice, y) for hh_xslice in hh], x)

    lkhds = [lkhd / _prob_mass(lkhd) for lkhd in lkhds]
    likelihood /= _prob_mass(likelihood)

    # Likelihood contours.
    def _plot_contours(hh, _alpha):

        # Get likelihood surface heights on grid and cumulative heights.
        h_flat = np.flip(np.sort(hh.flatten()))
        cum_h = np.cumsum(h_flat)
        cum_h /= cum_h[-1]

        # Determine likelihood surface height levels.
        h_levels = np.zeros_like(SIGMA_LEVELS)
        for n, quantile in enumerate(SIGMA_LEVELS):
            try:
                h_levels[n] = h_flat[cum_h <= quantile][-1]
            except IndexError:
                h_levels[n] = h_flat[0]

        # Plot the contour.
        try:
            contour = main_panel.contourf(
                xx, yy, hh, h_levels, antialiased=True,
                cmap=cmap, alpha=_alpha
            )
        except ValueError as error:
            if str(error) == "Contour levels must be increasing":
                raise ValueError(
                    "Cannot process likelihood values into contours."
                ) from error
            raise ValueError from error

        main_panel.contour(
            contour, colors=cmap(cmap.N), alpha=min(2 * _alpha, 1.)
        )

    _plot_contours(likelihood, alpha)
    if plt_scatter:
        for lkhd in lkhds:
            _plot_contours(lkhd, CONTOUR_SCATTER_RELATIVE_ALPHA * alpha)

    # Likelihood marginal PDFs.
    def _plot_pdfs(hh, _estimate, _alpha):

        pdf_x = np.asarray([simps(hh_xslice, y) for hh_xslice in hh])
        pdf_y = np.asarray([simps(hh_yslice, x) for hh_yslice in hh.T])
        cdf_x = cumtrapz(pdf_x, x, initial=0.)
        cdf_y = cumtrapz(pdf_y, y, initial=0.)

        pdf_x /= cdf_x[-1]
        pdf_y /= cdf_y[-1]
        cdf_x /= cdf_x[-1]
        cdf_y /= cdf_y[-1]

        x_panel.plot(x, pdf_x, c=cmap(cmap.N), alpha=_alpha)
        y_panel.plot(pdf_y, y, c=cmap(cmap.N), alpha=_alpha)

        if _estimate:
            if _estimate == 'maximum':
                x_fit_idx = np.argmax(pdf_x)
                y_fit_idx = np.argmax(pdf_y)
            elif _estimate == 'median':
                x_fit_idx = np.argmin(np.abs(cdf_x - 1./2.))
                y_fit_idx = np.argmin(np.abs(cdf_y - 1./2.))

            x_lower_idx = np.argmin(np.abs(cdf_x - ONE_SIGMA_QUANTILES[0]))
            x_upper_idx = np.argmin(np.abs(cdf_x - ONE_SIGMA_QUANTILES[-1]))

            y_lower_idx = np.argmin(np.abs(cdf_y - ONE_SIGMA_QUANTILES[0]))
            y_upper_idx = np.argmin(np.abs(cdf_y - ONE_SIGMA_QUANTILES[-1]))

            x_fit, x_lower, x_upper = x[[x_fit_idx, x_lower_idx, x_upper_idx]]
            y_fit, y_lower, y_upper = y[[y_fit_idx, y_lower_idx, y_upper_idx]]

            dx_lower, dx_upper = x_fit - x_lower, x_upper - x_fit
            dy_lower, dy_upper = y_fit - y_lower, y_upper - y_fit

            if x_precision is not None:
                x_fit = np.around(x_fit, decimals=x_precision)
                dx_lower = np.around(dx_lower, decimals=x_precision)
                dx_upper = np.around(dx_upper, decimals=x_precision)
                if x_precision == 0:
                    x_fit, dx_lower, dx_upper = \
                        map(int, (x_fit, dx_lower, dx_upper))

            if y_precision is not None:
                y_fit = np.around(y_fit, decimals=y_precision)
                dy_lower = np.around(dy_lower, decimals=y_precision)
                dy_upper = np.around(dy_upper, decimals=y_precision)
                if y_precision == 0:
                    y_fit, dy_lower, dy_upper = \
                        map(int, (y_fit, dy_lower, dy_upper))

            x_panel.vlines(
                x_fit, ymin=0., ymax=pdf_x[x_fit_idx],
                linestyles='--', colors=[cmap(cmap.N)],
                label=r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
                    x_label, x_fit, dx_lower, dx_upper
                )
            )
            x_panel.fill_between(
                x[x_lower_idx:(x_upper_idx + 1)],
                pdf_x[x_lower_idx:(x_upper_idx + 1)],
                color=[cmap(cmap.N)], antialiased=True, alpha=AREA_FILL_ALPHA
            )
            # x_panel.annotate(
            #     r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
            #         x_label, x_fit, dx_lower, dx_upper
            #     ),
            #     xy=(0.5, 1.1),
            #     xycoords='axes fraction',
            #     verticalalignment='baseline',
            #     horizontalalignment='center'
            # )

            y_panel.hlines(
                y_fit, xmin=0., xmax=pdf_y[y_fit_idx],
                linestyles='--', colors=[cmap(cmap.N)],
                label=r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
                    y_label, y_fit, dy_lower, dy_upper
                )
            )
            y_panel.fill_betweenx(
                y[y_lower_idx:(y_upper_idx + 1)],
                pdf_y[y_lower_idx:(y_upper_idx + 1)],
                color=[cmap(cmap.N)], antialiased=True, alpha=AREA_FILL_ALPHA
            )
            # y_panel.annotate(
            #     r"${} {{{}}}_{{-{}}}^{{+{}}}$".format(
            #         y_label, y_fit, dy_lower, dy_upper
            #     ),
            #     xy=(1.1, 0.5),
            #     xycoords='axes fraction',
            #     rotation=-90,
            #     verticalalignment='center',
            #     horizontalalignment='left'
            # )

            return (x_fit, x_lower, x_upper), (y_fit, y_lower, y_upper)

        return None, None

    x_result, y_result = _plot_pdfs(likelihood, _estimate=estimate, _alpha=1.)
    if plt_scatter:
        for lkhd in lkhds:
            _plot_pdfs(lkhd, _estimate=False, _alpha=LINE_SCATTER_ALPHA)

    if estimate:
        main_panel.scatter(
            x_result[0], y_result[0], marker='+', s=64, c=[cmap(cmap.N)],
            label=label
        )

    return fig, x_result, y_result


def plot_likelihood(*args, sample_points_x=None, sample_points_y=None,
                    range_x=(), range_y=(), precision_x=None, precision_y=None,
                    label_x='', label_y='', truth_x=None, truth_y=None,
                    estimate='median', scatter_plot=False, fig=None,
                    cmaps=None, alphas=None, labels=None):
    """Plot likelihood constraints from evaluation on a coordinate mesh.

    Parameters
    ----------
    *args : (sequence of) sequence of :class:`numpy.ndarray`
        Log-likelihood values at sample points.
    sample_points_x : (sequence of) :class:`numpy.ndarray`
        First (parameter) coordinate values (default is `None` but must
        be overriden).  If a sequence is passed, its length must be the
        same as that of `args`.
    sample_points_y : (sequence of) :class:`numpy.ndarray` *or None, optional*
        Second (parameter) coordinate values (default is `None` but must
        be overriden if likelihood values are evaluated on a 2-d grid).
        If a sequence is passed, its length must be the same as that of
        `args`.
    range_x : tuple, optional
        Renormalisation range for the first (parameter) coordinate
        (default is ()).
    range_y : tuple, optional
        Renormalisation range for the second (parameter) coordinate
        (default is ()).
    precision_x : int or None, optional
        Precision for the first parameter estimate (default is `None`).
    precision_y : int or None, optional
        Precision for the second parameter estimate (default is `None`).
    label_x : str, optional
        x-axis parameter name as a TeX string (default is '').
    label_y : str, optional
        y-axis parameter name as a TeX string (default is '').
    truth_x : int or None, optional
        Truth value for the first parameter (default is `None`).
    truth_y : int or None, optional
        Truth value for the second parameter (default is `None`).
    estimate : {'maximum', 'median', None}, optional
        Parameter estimate type (default is 'median').
    scatter_plot : bool, optional
        Plot individual log-likelihood evaluations on coordinate meshes
        (default is `False`).
    fig : :class:`matplotlib.figure.Figure` *or None, optional*
        Any existing figures to plot on (default is `None`).
    label : (sequence of) str or None, optional
        Label for the likelihood result (default is `None`).  If a
        sequence is passed, its length must be the same as that of `args`.
    cmap : (sequence of) :class:`matplotlib.ScalarMappable` or None, optional
        Colour map(s) for sampled likelihood contours (default is `None`).
        If a sequence is passed, its length must be the same as that
        of `args`.
    alpha : (sequence of) float or None, optional
        Transparency alpha(s) for sampled likelihood contours (default is
        `None`).  If a sequence is passed, its length must be the same as
        that of `args`.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` or None, optional
        Any existing figures to plot on.
    x_results : list of tuple
        Best estimate, lower uncertainty and upper uncertainty of the
        first parameter for each likelihood value sets.
    y_results : list of tuple
        Best estimate, lower uncertainty and upper uncertainty of the
        second parameter for each likelihood value sets.

    """
    # Check input likelihood dimensions.
    likeliood_sets, ndim = [], []
    for arg in args:
        if isinstance(arg, Sequence):
            likeliood_sets.append(arg)
            ndim.extend(list(map(np.ndim, arg)))
        else:
            likeliood_sets.append([arg])
            ndim.append(np.ndim(arg))
    if len(set(ndim)) > 1:
        raise ValueError(
            "Input likelihood results have inconsistent dimensions."
        )
    ndim = ndim[0]

    if labels is None:
        labels = [None] * len(args)
    elif isinstance(labels, str):
        labels = [labels]

    if ndim == 1:
        fig = fig or plt.figure()
        if not fig.axes:
            fig.add_subplot(111)
        canvas = fig.axes[0]

        x_results = []
        for likelihood_results, label in zip(likeliood_sets, labels):
            fig, x_result = _plot_likelihood_1d(
                fig, likelihood_results, sample_points_x,
                x_precision=precision_x, x_range=range_x, x_label=label_x,
                label=label, estimate=estimate, plt_scatter=scatter_plot
            )
            x_results.append(x_result)

        if truth_x is not None:
            canvas.axvline(
                x=truth_x, ls='--', label="truth {}".format(truth_x)
            )

        canvas.set_xlim(sample_points_x.min(), sample_points_x.max())
        canvas.set_xlabel(r'${}$'.format(label_x))

        if labels == [None] * len(args):
            canvas.legend()

        return fig, x_results

    if ndim == 2:
        if cmaps is None:
            cmaps = ['Oranges', 'Greens', 'Blues'] \
                + ['Blues'] * (len(args) - 3)
        elif not isinstance(cmaps, Sequence):
            cmaps = [cmaps]
        cmaps = list(map(
            lambda name: ListedColormap(sns.color_palette(name)), cmaps
        ))

        if alphas is None:
            alphas = [1.] * len(args)

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

        x_results, y_results = [], []
        for likelihood_results, cmap, alpha, label in zip(
                likeliood_sets, cmaps, alphas, labels
            ):
            fig, x_result, y_result = _plot_likelihood_2d(
                fig, cmap, alpha, likelihood_results,
                sample_points_x, sample_points_y,
                x_label=label_x, y_label=label_y,
                x_precision=precision_x, y_precision=precision_y,
                x_range=range_x, y_range=range_y,
                label=label, estimate=estimate, plt_scatter=scatter_plot
            )
            x_results.append(x_result)
            y_results.append(y_result)

        if truth_x is not None:
            main_panel.axvline(truth_x, c='k', ls='--')
        if truth_y is not None:
            main_panel.axhline(truth_y, c='k', ls='--')

        main_panel.set_xlim(sample_points_x.min(), sample_points_x.max())
        main_panel.set_ylim(sample_points_y.min(), sample_points_y.max())
        main_panel.set_xlabel(r'${}$'.format(label_x))
        main_panel.set_ylabel(r'${}$'.format(label_y))

        if labels != [None] * len(args):
            main_panel.legend(handletextpad=0)

        x_panel.legend()
        y_panel.legend()

        main_panel.axes.tick_params(
            axis='x', which='both', direction='in', top=True
        )
        main_panel.axes.tick_params(
            axis='y', which='both', direction='in', right=True
        )

        x_panel.set_ylim(bottom=0)
        x_panel.axes.tick_params(
            axis='x', which='both', top=False, bottom=False, labelbottom=False
        )
        x_panel.axes.tick_params(
            axis='y', which='both', left=False, right=False, labelleft=False
        )
        x_panel.spines['top'].set_visible(False)
        x_panel.spines['left'].set_visible(False)
        x_panel.spines['right'].set_visible(False)

        y_panel.set_xlim(left=0)
        y_panel.axes.tick_params(
            axis='x', which='both', top=False, bottom=False, labelbottom=False
        )
        y_panel.axes.tick_params(
            axis='y', which='both', left=False, right=False, labelleft=False
        )
        y_panel.spines['top'].set_visible(False)
        y_panel.spines['bottom'].set_visible(False)
        y_panel.spines['right'].set_visible(False)

        return fig, x_results, y_results

    raise ValueError(
        "Input likelihood results have dimensions greater than 2."
    )


def process_likelihood_results():
    """Collate likelihood results from the input directory with the
    specified filename pattern.

    Returns
    -------

    """
    search_filename = str(
        input_dir/input_file.replace("=[", "=[[]").replace("],", "[]],")
    )

    lkhds, fnls, b_1s = [], [], []
    for matched_file in glob.iglob(search_filename):
        matched_file_data = np.load(matched_file)
        lkhds.append(sum(
            (
                matched_file_data[var].T
                for var in matched_file_data.files if 'likelihood' in var
            )
        ))
        fnls.append(matched_file_data['f_nl'])
        b_1s.append(matched_file_data['b_1'])

    if not all(map(np.array_equal, *tee(fnls))) \
            or not all(map(np.array_equal, *tee(b_1s))):
        warnings.warn(
            "Input likelihoods are not evaludated on the same parameter grid."
        )

    fnls, b_1s = np.average(fnls, axis=0), np.average(b_1s, axis=0)

    return lkhds, fnls, b_1s


if __name__ == "__main__":

    NG = 0.
    MAP = "hybrid"
    MASK_TAG = "1.0"  # random0_BOSS_DR12v5_CMASS_North
    SELECTION_TAG = "None"  # [100.0,500.0]

    SCALE_TAG = "[None,0.04,0.1]"
    ORDER_TAG = "[0]"

    DEGREE_TAG = "False"
    RSD_TAG = "False"

    # Set I/O paths.
    input_dir = data_dir/"raw"/"likelihoods"
    input_file = "likelihood-({}).npz".format(",".join([
        "source=halo-(NG={}.,z=1.)-{}".format(int(NG), "*"),
        "map={}".format(MAP), "scale={}".format(SCALE_TAG),
        "orders={}".format(ORDER_TAG), "excl_monop={}".format(DEGREE_TAG),
        "rsd={}".format(RSD_TAG),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))

    output_dir = data_dir/"processed"/"likelihoods"
    output_filename = "likelihood-({})".format(",".join([
        "source=halo-(NG={}.,z=1.)".format(int(NG)),
        "map={}".format(MAP), "scale={}".format(SCALE_TAG),
        "orders={}".format(ORDER_TAG), "excl_monop={}".format(DEGREE_TAG),
        "rsd={}".format(RSD_TAG),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))
    output_file = output_dir/output_filename

    confirm_directory(output_dir)

    # Collate and export likelihoods.
    likelihoods, f_nl_coords, b_1_coords = process_likelihood_results()
    output_path = output_file.with_suffix(output_file.suffix + '.npz')
    if overwrite_protection(output_path):
        np.savez(
            output_file,
            likelihoods=likelihoods,
            f_nl_coords=f_nl_coords,
            b_1_coords=b_1_coords
        )
    # pylint: disable=using-constant-test
    if False:
        MAP = "cartesian"
        SCALE_TAG = "[None,None,0.1]"
        output_filename = "likelihood-({})".format(",".join([
            "source=halo-(NG={}.,z=1.)".format(int(NG)),
            "map={}".format(MAP), "scale={}".format(SCALE_TAG),
            "orders={}".format(ORDER_TAG), "excl_monop={}".format(DEGREE_TAG),
            "rsd={}".format(RSD_TAG),
            "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
        ]))
        output_file = output_dir/output_filename

        loaded_results = np.load(
            output_file.with_suffix(output_file.suffix + '.npz')
        )
        for name in loaded_results.files:
            globals()[name] = loaded_results[name]

    # Plot constraints.
    figure, *results = plot_likelihood(
        likelihoods, sample_points_x=f_nl_coords, sample_points_y=b_1_coords,
        label_x=r'f_{\mathrm{NL}}', label_y=r'b_1',
        precision_x=0, precision_y=2, truth_x=NG, # estimate='maximum',
        scatter_plot=True, # alphas=[1.0, 0.33]
    )
    # pylint: disable=using-constant-test
    if True:
        figure.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))
