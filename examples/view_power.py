"""Visualise recovered power spectrum.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline

from harmonia.collections import harmony

# Default plotting options.
CLOSE_ALL = True

MARKER = '+'
ANNO_SIZE = 6

TRANSPARENCY_1, TRANSPARENCY_2 = 1/4, 1/16
ERROR_CANVAS_GRID, ERROR_ROWSPAN = (4, 8), 3

COLOUR = {'ref': '#C40233', 'def': '#0087BD', 'add': '#009F6B',}
LABEL = {'ref': 'Cartesian', 'def': 'spherical', 'add': 'spherical',}
TITLE = {'def': None, 'add': None}


def view_spectrum(data, case='error'):

    plt.style.use(harmony)
    if CLOSE_ALL:
        plt.close('all')

    if case == 'single':
        fig = plt.figure()

        plt.errorbar(
            data['k'], data['Pk'],
            xerr=data['dk']/np.sqrt(data['dof1']),
            yerr=data['dPk']/np.sqrt(data['dof2']),
            elinewidth=0.8, color=COLOUR['ref'], label=LABEL['ref']
            )

        plt.loglog(
            data['kln'], data['Pln'], color=COLOUR['def'], label=LABEL['def']
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + data['dPln']/np.sqrt(data['dof2']),
            facecolor=COLOUR['def'], alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - 2*data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + 2*data['dPln']/np.sqrt(data['dof2']),
            facecolor=COLOUR['def'], alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data['kln'][idx], data['Pln'][idx],
                    marker=MARKER, color=COLOUR['def']
                    )
                plt.annotate(
                    r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                    xy=(data['kln'][idx], data['Pln'][idx]),
                    verticalalignment='bottom', fontsize=ANNO_SIZE
                    )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.legend()

    if case == 'error':
        fig = plt.figure(figsize=(7, 5.5))

        plt.subplot2grid(
            ERROR_CANVAS_GRID, (0, 0),
            rowspan=ERROR_ROWSPAN, colspan=ERROR_CANVAS_GRID[1]
            )

        plt.errorbar(
            data['k'], data['Pk'],
            xerr=data['dk']/np.sqrt(data['dof1']),
            yerr=data['dPk']/np.sqrt(data['dof2']),
            elinewidth=0.8, color=COLOUR['ref'], label=LABEL['ref']
            )

        plt.loglog(
            data['kln'], data['Pln'], color=COLOUR['def'], label=LABEL['def']
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + data['dPln']/np.sqrt(data['dof2']),
            facecolor=COLOUR['def'], alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - 2*data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + 2*data['dPln']/np.sqrt(data['dof2']),
            facecolor=COLOUR['def'], alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data['kln'][idx], data['Pln'][idx],
                    marker=MARKER, color=COLOUR['def']
                    )
                plt.annotate(
                    r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                    xy=(data['kln'][idx], data['Pln'][idx]),
                    verticalalignment='bottom', fontsize=ANNO_SIZE
                    )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.legend()

        plt.subplot2grid(
            ERROR_CANVAS_GRID, (ERROR_ROWSPAN, 0),
            rowspan=ERROR_CANVAS_GRID[0]-ERROR_ROWSPAN,
            colspan=ERROR_CANVAS_GRID[1]
            )

        cart = IUSpline(data['k'], data['Pk'])(data['kln'])

        plt.plot(data['kln'], data['Pln']/cart-1, '--k')
        plt.axhline(y=0., c='k', alpha=.5)

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$\Delta P(k)/P_\textrm{Cart}(k)$')

    plt.subplots_adjust(hspace=0, wspace=0)

    return fig
