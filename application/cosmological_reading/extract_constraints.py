"""Extract cosmological constraints from sampled log-likelihood values.

"""
import glob
import os
import sys
import warnings
from collections import defaultdict
from collections.abc import Sequence
from itertools import tee

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata

try:
    from application import confirm_directory, overwrite_protection
    from application import data_dir, harmony
except ImportError:
    # Adds to Python search path.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../"
    ))
    from application import confirm_directory, overwrite_protection
    from application import data_dir, harmony

from plot_constraints import plot_1d_constraints, plot_2d_constraints

sns.set(style='ticks', font='serif')
plt.style.use(harmony)

legend_state = ([], [])


def plot_likelihood(*args, sample_points_x=None, sample_points_y=None,
                    range_x=(), range_y=(), precision_x=None, precision_y=None,
                    label_x='', label_y='', truth_x=None, truth_y=None,
                    estimate='median', scatter_plot=False, aggregate=False,
                    fig=None, label=None, colour=None, cmap=None, alpha=None):
    """Plot likelihood evaluation results on a parameter grid.

    Parameters
    ----------
    *args : (sequence of) :class:`numpy.ndarray`
        Log-posterior values at sample points.
    sample_points_x : (sequence of) :class:`numpy.ndarray`
        First parameter values (default is `None` but must be overriden).
        If a sequence is passed, its length must be the same as that of
        `args`.
    sample_points_y : (sequence of) :class:`numpy.ndarray` *or None, optional*
        Second parameter values (default is `None` but must be overriden
        if likelihood values are evaluated on a 2-d grid).  If a sequence
        is passed, its length must be the same as that of `args`.
    range_x : tuple, optional
        Renormalisation range for the first parameter (default is ()).
    range_y : tuple, optional
        Renormalisation range for the second parameter (default is ()).
    precision_x : int or None, optional
        Precision for the first parameter estimate (default is `None`).
    precision_y : int or None, optional
        Precision for the second parameter estimate (default is `None`).
    label_x : str, optional
        *x*-axis parameter name as a TeX string (default is '').
    label_y : str, optional
        *y*-axis parameter name as a TeX string (default is '').
    truth_x : float or None, optional
        Truth value for the first parameter (default is `None`).
    truth_y : float or None, optional
        Truth value for the second parameter (default is `None`).
    estimate : {'maximum', 'median', None}, optional
        Parameter estimate type (default is 'median').
    scatter_plot : bool, optional
        Plot individual log-likelihood evaluations on parameter grids
        (default is `False`).
    fig : :class:`matplotlib.figure.Figure` *or None, optional*
        Any existing figures to plot on (default is `None`).
    aggregate : bool, optional
        Whether to aggregate likelihood evaluations by summing logarithmic
        likelihood values (default is `False`).
    label : (sequence of) str or None, optional
        Label for the likelihood result (default is `None`).  If a
        sequence is passed, its length must be the same as that of `args`.
    colour : (sequence of) str or None, optional
        Colour for sampled likelihood (default is `None`).  If a sequence
        is passed, its length must be the same as that of `args`.
    cmap : (sequence of) :class:`matplotlib.ScalarMappable` or None, optional
        Colour map for sampled likelihood contours (default is `None`).
        If a sequence is passed, its length must be the same as that
        of `args`.
    alpha : (sequence of) float or None, optional
        Transparency value for sampled likelihood contours (default is
        `None`).  If a sequence is passed, its length must be the same as
        that of `args`.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Any existing figures to plot on.
    x_estimate, y_estimate : list of tuple
        Parameter estimate, lower uncertainty and upper uncertainty of
        for each likelihood value sets.

    """
    # pylint: disable=redefined-argument-from-local

    LS = iter(['-', '--', '-.'])

    # Check likelihood dimensions and restructure input arrays.
    likelihood_sets, ndim = [], []
    for arg in args:
        if isinstance(arg, Sequence):
            likelihood_sets.append(arg)
            ndim.extend(list(map(np.ndim, arg)))
        else:
            likelihood_sets.append([arg])
            ndim.append(np.ndim(arg))

    if len(set(ndim)) == 1:
        ndim = ndim[0]
    else:
        raise ValueError(
            "Input likelihood results have inconsistent dimensions."
        )

    # Set sample points, label, colour map and transparency values.
    if isinstance(sample_points_x, Sequence):
        sample_points_xs = sample_points_x
    else:
        sample_points_xs = [sample_points_x]

    if isinstance(sample_points_y, Sequence):
        sample_points_ys = sample_points_y
    else:
        sample_points_ys = [sample_points_y]

    if label is None:
        labels = [None] * len(args)
    elif isinstance(label, str):
        labels = [label]
    else:
        labels = label

    if colour is None:
        colours = [None] * len(args)
    elif isinstance(colour, str):
        colours = [cmap]
    else:
        colours = colour

    if cmap is None:
        cmaps = [None] * len(args)
    elif isinstance(cmap, str):
        cmaps = [cmap]
    else:
        cmaps = cmap

    if alpha is None:
        alphas = [1./2.] * len(args)
    elif isinstance(alpha, float):
        alphas = [alpha]
    else:
        alphas = alpha

    # Plot constraints.
    aggregation = 'combine' if aggregate else 'average'

    if ndim == 1:
        if fig is None:
            fig, canvas = plt.subplots()
        else:
            canvas, *_ = fig.axes

        x_estimate = []
        # pylint: disable=redefined-outer-name
        for likelihoods, x, label, colour \
                in zip(likelihood_sets, sample_points_xs, labels, colours):
            fig, x_estimate_ = plot_1d_constraints(
                likelihoods, x, fig=fig, label=label, colour=colour,
                x_label=label_x, x_range=range_x, aggregation=aggregation,
                estimation=estimate, precision=precision_x,
                scatter_plot=scatter_plot
            )
            x_estimate.append(x_estimate_)

        if truth_x is not None:
            canvas.axvline(x=truth_x, ls='--', zorder=3)

        canvas.set_xlim(np.min(sample_points_xs), np.max(sample_points_xs))
        canvas.set_ylim(bottom=0.)
        canvas.set_xlabel(r'${}$'.format(label_x))
        canvas.legend()

        return fig, x_estimate

    if ndim == 2:
        # pylint: disable=global-statement
        global legend_state
        if fig is None:
            fig = plt.figure(figsize=(5, 5))
            main_panel = plt.subplot2grid(
                (4, 4), (1, 0), rowspan=3, colspan=3
            )
            x_panel = plt.subplot2grid(
                (4, 4), (0, 0), colspan=3, sharex=main_panel
            )
            y_panel = plt.subplot2grid(
                (4, 4), (1, 3), rowspan=3, sharey=main_panel
            )
            legend_state = ([], [])
        else:
            main_panel, x_panel, y_panel = fig.axes

        cmaps = list(map(
            lambda name: ListedColormap(sns.color_palette(name)), cmaps
        ))

        x_estimate, y_estimate = [], []
        for likelihoods, x, y, label, cmap, alpha in zip(
                likelihood_sets, sample_points_xs, sample_points_ys,
                labels, cmaps, alphas
            ):
            fig, x_estimate_, y_estimate_, patch = plot_2d_constraints(
                likelihoods, x, y, fig=fig, cmap=cmap, alpha=alpha,
                x_label=label_x, y_label=label_y,
                x_range=range_x, y_range=range_y,
                x_precision=precision_x, y_precision=precision_y,
                aggregation=aggregation, estimation=estimate,
                scatter_plot=scatter_plot, line_style=next(LS)
            )
            x_estimate.append(x_estimate_)
            y_estimate.append(y_estimate_)

            legend_state[0].append(patch)
            legend_state[1].append(label)


        if truth_x is not None:
            main_panel.axvline(truth_x, c='k', ls='--', zorder=3)
        if truth_y is not None:
            main_panel.axhline(truth_y, c='k', ls='--', zorder=3)

        main_panel.set_xlim(
            min(map(np.min, sample_points_xs)),
            max(map(np.max, sample_points_xs))
        )
        main_panel.set_ylim(
            min(map(np.min, sample_points_ys)),
            max(map(np.max, sample_points_ys))
        )
        main_panel.set_xlabel(r'${}$'.format(label_x))
        main_panel.set_ylabel(r'${}$'.format(label_y))

        if labels != [None] * len(args):
            main_panel.legend(*legend_state, handlelength=1.6)
        x_panel.legend(loc='upper left')
        y_panel.legend(loc='upper left')

        # Tidy up panels.
        main_panel.xaxis.set_minor_locator(AutoMinorLocator())
        main_panel.yaxis.set_minor_locator(AutoMinorLocator())
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

        return fig, x_estimate, y_estimate

    raise ValueError(
        "Input likelihood results have dimensions greater than 2."
    )


def aggregate_likelihood_results(constituent='all', load=False, export=False,
                                 x_name=None, y_name=None,
                                 x_grid=None, y_grid=None):
    """Aggregate likelihood results.

    Parameters
    ----------
    constituent : {'all', 'spherical', 'cartesian'}, optional
        Likelihood constituent (default is 'all').
    load : bool, optional
        If `True` (default is `False`), load aggregated results.
    export : bool, optional
        If `True` (default is `False`), save aggregated results.
    x_name, y_name : str or None, optional
        Name of the parameter (default is `None`).
    x_grid, y_grid : :class:`numpy.ndarray` *or None, optional*
        Parameter 1-d grid to interpolate the likelihood results.

    Returns
    -------
    likelihood_grids, x_coords, y_coords : float :class:`numpy.ndarray`
        Likelihood and coordinate grids.

    """
    if load:
        # Load collated likelihood data from file.
        loaded_data = np.load(
            output_file.with_suffix(output_file.suffix + '.npz')
        )
        collated_data = {
            data_name: loaded_data[data_name]
            for data_name in loaded_data.files
        }
    else:
        # Collate likelihood data from files.
        search_filename = str(
            input_dir/input_file.replace("=[", "=[[]").replace("],", "[]],")
        )

        matched_files = glob.glob(search_filename)
        if not matched_files:
            raise IOError("No files matched input information.")

        collated_data = defaultdict(list)
        for matched_file in matched_files:
            matched_file_data = np.load(matched_file)
            for data_name in matched_file_data.files:
                collated_data[data_name].append(matched_file_data[data_name])

        # Save collated likelihood data to file.
        output_path = output_file.with_suffix(output_file.suffix + '.npz')
        if export and overwrite_protection(output_path):
            np.savez(collated_data, **collated_data)

    # Extract required likelihood and parameter data.
    likelihood_name = 'likelihood' if constituent == 'all' \
        else constituent + '_likelihood'

    likelihood_constituents = [
        likelihood_data
        for data_name, likelihood_data in collated_data.items()
        if likelihood_name in data_name
    ]

    likelihoods_ = [
        np.real_if_close(np.squeeze(likelihood), tol=10**10)
        for likelihood in list(map(sum, zip(*likelihood_constituents)))
    ]

    x_coords = collated_data[x_name]
    y_coords = collated_data[y_name]

    # Return either uniterpolated or interpolated grid data.
    if x_grid is None and y_grid is None:
        if not all(map(np.array_equal, *tee(x_coords))) \
                or not all(map(np.array_equal, *tee(y_coords))):
            warnings.warn(
                "Input likelihoods are not evaluated "
                "on the same parameter grids."
            )

        x_coords = np.mean(x_coords, axis=0)
        y_coords = np.mean(y_coords, axis=0)
        likelihood_grids = [likelihood.T for likelihood in likelihoods_]

        return likelihood_grids, x_coords, y_coords

    xx, yy = np.meshgrid(x_grid, y_grid)

    likelihood_grids = []
    for likelihood, x, y in zip(likelihoods_, x_coords, y_coords):
        likelihood_flat = likelihood.flatten()
        xy_flat = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        likelihood_grids.append(griddata(
            xy_flat, likelihood_flat, (xx.T, yy.T), method='cubic'
        ))

    return likelihood_grids, x_grid, y_grid


TRACER = "halos"
SERIES = "-cut_2"

if __name__ == "__main__":

    MAP = "hybrid"
    NG = 0.

    MASK_TAG = "1.0"  # "random0_BOSS_DR12v5_CMASS_North"  #
    SELECTION_TAG = "None"  # "[100.0,500.0]"  #

    SCALE_TAG = "[None,0.04,0.09]"
    ORDER_TAG = "[0]"

    DEGREE_TAG = "False"
    RSD_TAG = "False"

    # Set I/O paths.
    input_dir = data_dir/"raw"/"likelihoods"
    input_file = "likelihood-({}).npz".format(",".join([
        "source={}-(NG={}.,z=1.){}-{}".format(TRACER, int(NG), SERIES, "*"),
        "map={}".format(MAP), "scale={}".format(SCALE_TAG),
        "orders={}".format(ORDER_TAG), "excl_monop={}".format(DEGREE_TAG),
        "rsd={}".format(RSD_TAG),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))

    output_dir = data_dir/"processed"/"likelihoods"
    output_filename = "likelihood-({})".format(",".join([
        "source={}-(NG={}.,z=1.){}".format(TRACER, int(NG), SERIES),
        "map={}".format(MAP), "scale={}".format(SCALE_TAG),
        "orders={}".format(ORDER_TAG), "excl_monop={}".format(DEGREE_TAG),
        "rsd={}".format(RSD_TAG),
        "mask={}".format(MASK_TAG), "selection={}".format(SELECTION_TAG),
    ]))
    output_file = output_dir/output_filename

    confirm_directory(output_dir)

    # Aggregate likelihood results.
    f_nl_grid = np.linspace(-300, 300, num=1201)
    b_1_grid = np.linspace(2.1, 2.6, num=201)

    likelihoods, f_nl_coords, b_1_coords = aggregate_likelihood_results(
        constituent='all', x_name='f_nl', y_name='b_1',
        # x_grid=f_nl_grid, y_grid=b_1_grid
    )

    # Plot constraints.
    figure, *results = plot_likelihood(
        likelihoods,
        sample_points_x=[f_nl_coords,],
        sample_points_y=[b_1_coords,],
        label_x=r'f_{\mathrm{NL}}',
        label_y=r'b_1',
        precision_x=0,
        precision_y=3,
        # estimate='maximum',
        # aggregate=True,
        truth_x=NG,
        scatter_plot=True,
    )
    # pylint: disable=using-constant-test
    if False:
        figure.savefig(output_file.with_suffix(output_file.suffix + '.pdf'))
