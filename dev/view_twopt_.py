"""Visualise recovered 2-point function values.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from harmonia.collections import harmony


def view_covariance_(d2pt, m2pt=None, diff=False, diag=None, lb=None, ind=None):  # !!!

    plt.style.use(harmony)
    plt.close('all')
    sns.set(style='ticks', font='serif')

    if ind is not None:
        d2pt = d2pt[ind, ind]
        if m2pt is not None: m2pt = m2pt[ind, ind]

    if lb is None: lb = 50

    if m2pt is None:
        if diag == 'off': np.fill_diagonal(d2pt, np.nan)
        sns.heatmap(
            d2pt, center=0, square=True, xticklabels=lb, yticklabels=lb
            )
        return

    if diff:
        if diag == 'only':
            mask = np.diag(np.ones(len(d2pt)))
        else:
            mask0 = ~np.isclose(m2pt, 0)
            mask1 = m2pt / d2pt > 1e-3
            mask = np.logical_and(mask0, mask1)

        diff = np.where(mask, m2pt/d2pt, np.nan)  # m2pt/d2pt, d2pt/m2pt
        sns.heatmap(
            diff, center=1, annot=False,# vmin=0.9, vmax=1.1,
            square=True, xticklabels=lb, yticklabels=lb
            )
        return diff

    if diag == 'off':
        np.fill_diagonal(d2pt, np.nan)
        np.fill_diagonal(m2pt, np.nan)
    if diag == 'only':
        mask = np.diag(np.ones(len(d2pt)))
        d2pt = np.where(mask, d2pt, np.nan)
        m2pt = np.where(mask, m2pt, np.nan)

    plt.subplot2grid((4, 9), (0, 0), rowspan=4, colspan=4)
    sns.heatmap(
        d2pt, center=0,
        square=True, xticklabels=lb, yticklabels=lb, cbar_kws={'shrink': 0.6}
        )

    plt.subplot2grid((4, 9), (0, 5), rowspan=4, colspan=4)
    sns.heatmap(
        m2pt, center=0,
        square=True, xticklabels=lb, yticklabels=lb, cbar_kws={'shrink': 0.6}
        )

    return
