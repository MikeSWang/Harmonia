"""Visualise recovered 2-point function values.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def view_covariance(data, model=None, ratio=False, diag=False, select_idx=None,
                    tick_labels=None):
    """View 2-point covariance matrices.

    Parameters
    ----------
    data : array_like
        Data 2-point covariance.
    model : array_like
        Model 2-point covariance.
    ratio : {False, 'd2m', 'm2d'}, optional
        If ``'off'`` (default), this has no effect.  If ``'d2m'`` or ``'m2d'``,
        plot the ratio of `data` to `model` or vice versa.
    diag : {False, 'off', 'only'}, optional
        Whether the diagonal elements should masked (``'off'``), or be the only
        non-masked ones (``'only'``), or no masking is applied (`False`,
        default).
    select_idx : slice or None, optional
        Index slice selecting relevant data entries (default is `None`).
    tick_labels : list-like, int or None, optional
        Tick labels (default is `None`).

    Returns
    -------
    fig : :class:`matplotlib.Figure`
        Plotted figure.

    """
    TICK_INTERVAL = 25
    HEATMAP_OPT = dict(
        square=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )

    sns.set(style='ticks', font='serif')
    fig = plt.figure("Two-point covariance")

    if tick_labels is None:
        tick_labels = TICK_INTERVAL
    if select_idx is not None:
        data = data[select_idx, select_idx]
        if model is not None:
            model = model[select_idx, select_idx]

    if diag:
        if diag == 'only':
            diag_mask = np.eye(len(data))
        elif diag == 'off':
            diag_mask = 1 - np.eye(len(data))
        data = np.where(diag_mask, data, np.nan)
        if model is not None:
            model = np.where(diag_mask, model, np.nan)

    if ratio:
        global ratio_array
        with np.errstate(invalid='ignore', divide='ignore'):
            safety_mask = np.logical_and(
                ~np.isclose(model, 0),
                model/data > 1e-3,
            )
            if ratio == 'd2m':
                ratio_array = np.where(safety_mask, data/model, np.nan)
            elif ratio == 'm2d':
                ratio_array = np.where(safety_mask, model/data, np.nan)

        sns.heatmap(ratio_array, center=1, **HEATMAP_OPT)

    if model is None:
        sns.heatmap(data, center=0, **HEATMAP_OPT)
    else:
        plt.subplot2grid((4, 9), (0, 0), rowspan=4, colspan=4)
        sns.heatmap(data, center=0, cbar_kws={'shrink': 0.6}, **HEATMAP_OPT)

        plt.subplot2grid((4, 9), (0, 5), rowspan=4, colspan=4)
        sns.heatmap(model, center=0, cbar_kws={'shrink': 0.6}, **HEATMAP_OPT)

    return fig
