"""Visualise recovered 2-point function values.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from harmonia.collections import harmony


def view_covariance(d2pt, m2pt=None, indices=None, label_interval=None,
                    diff=False, diag='none'):

    plt.style.use(harmony)
    plt.close('all')
    sns.set(style='ticks', font='serif')

    anno = False
    if label_interval is None:
        label_interval = 50
    if indices is not None:
        d2pt = d2pt[indices, indices]
        if m2pt is not None:
            m2pt = m2pt[indices, indices]

    if diag == 'only':
        mask = np.eye(len(d2pt))
        d2pt = np.where(mask, d2pt, np.nan)
        if m2pt is not None:
            m2pt = np.where(mask, m2pt, np.nan)
    elif diag == 'off':
        np.fill_diagonal(d2pt, np.nan)
        if m2pt is not None:
            np.fill_diagonal(m2pt, np.nan)

    if diff:
        with np.errstate(invalid='ignore', divide='ignore'):
            safety = np.logical_and(~np.isclose(m2pt, 0), m2pt/d2pt > 1e-3)
            diff = np.where(safety, m2pt/d2pt, np.nan)
        sns.heatmap(
            diff, center=1, annot=anno,
            square=True, xticklabels=label_interval, yticklabels=label_interval
            )
        return diff

    if m2pt is None:
        sns.heatmap(
            d2pt, center=0, square=True,
            xticklabels=label_interval, yticklabels=label_interval
            )
    else:
        plt.subplot2grid((4, 9), (0, 0), rowspan=4, colspan=4)
        sns.heatmap(
            d2pt, center=0, square=True, cbar_kws={'shrink': 0.6},
            xticklabels=label_interval, yticklabels=label_interval
            )

        plt.subplot2grid((4, 9), (0, 5), rowspan=4, colspan=4)
        sns.heatmap(
            m2pt, center=0, square=True, cbar_kws={'shrink': 0.6},
            xticklabels=label_interval, yticklabels=label_interval
            )

    return
