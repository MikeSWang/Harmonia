"""Visualise recovered power spectra.

"""
from __future__ import division

from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline

ANNO_SIZE = 6.
MARKER_SIZE = 3.
COLOUR = {
    'reference': '#C40233',
    'default': '#0087BD',
    'additional': '#009F6B',
}
ERROR_CANVAS_GRID = (4, 8)
ERROR_ROW_SPAN = 3
LABEL = {'reference': 'Cartesian', 'default': 'spherical'}
MARKER = '+'
SHADE_TRANSPARENCY = 1/5
BAR_TRANSPARENCY = 0.55
ERROR_LINEWIDTH = 2.
ERROR_PANEL_HT = 0.10
ERROR_PATCH_HT = 0.02


def view_spectrum(data, case='error', smoothed_data=None, error_sty='shade'):
    """Plot power spectra.

    Parameters
    ----------
    data : dict
        Power spectrum data.
    case : {'single', 'error'}, optional
        Plotting case, either a single plot (``'single'``) or with error
        panel (``'error'``).
    smoothed_data : dict or None, optional
        Smoothed power spectrum data.  If not `None` (default), this is
        plotted on top of plotted `data`.
    error_sty : {'shade', 'bar'}, optional
        Errorbar style, either 'bar' for bars or 'shade' (default) for
        shaded regions.

    """
    if data['Pk'][0] == 0.:
        for var, val in data.items():
            if var.endswith("k"):
                data[var] = val[1:]

    if case == 'single':

        fig = plt.figure()

        if error_sty == 'shade':
            plt.loglog(
                data['kln'],
                data['Pln'],
                color=COLOUR['default'],
                label=LABEL['default']
            )
            for layer in [1, 2]:
                plt.fill_between(
                    data['kln'],
                    data['Pln'] - layer * data['dPln'],
                    data['Pln'] + layer * data['dPln'],
                    facecolor=COLOUR['default'],
                    alpha=SHADE_TRANSPARENCY**layer
                )
        elif error_sty == 'bar':
            plt.errorbar(
                data['kln'],
                data['Pln'],
                1 * data['dPln'],
                fmt='o',
                markersize=MARKER_SIZE,
                color=COLOUR['default'],
                capsize=0.,
                elinewidth=ERROR_LINEWIDTH,
                ecolor=COLOUR['default'],
                alpha=BAR_TRANSPARENCY**1,
                label=LABEL['default']
            )
            plt.errorbar(
                data['kln'],
                data['Pln'],
                2 * data['dPln'],
                fmt='o',
                markersize=MARKER_SIZE,
                color=COLOUR['default'],
                capsize=0.,
                elinewidth=ERROR_LINEWIDTH,
                ecolor=COLOUR['default'],
                alpha=BAR_TRANSPARENCY**2
            )

        plt.errorbar(
            data['k'],
            data['Pk'],
            xerr=data['dk'],
            yerr=data['dPk'],
            color=COLOUR['reference'],
            label=LABEL['reference']
        )

        for idx, dbl_indices in enumerate(data['ln']):
            if dbl_indices[0] == 0:
                plt.scatter(
                    data['kln'][idx],
                    data['Pln'][idx],
                    marker=MARKER,
                    color=COLOUR['default']
                )
                plt.annotate(
                    str(dbl_indices),
                    xy=(data['kln'][idx], data['Pln'][idx]),
                    verticalalignment='bottom',
                    fontsize=ANNO_SIZE
                )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.legend()

    if case == 'error':

        fig = plt.figure(figsize=(8, 6))

        plt.subplot2grid(
            ERROR_CANVAS_GRID,
            (0, 0),
            rowspan=ERROR_ROW_SPAN,
            colspan=ERROR_CANVAS_GRID[1]
        )

        if error_sty == 'shade':
            plt.loglog(
                data['kln'],
                data['Pln'],
                color=COLOUR['default'],
                label=LABEL['default']
            )
            for layer in [1, 2]:
                plt.fill_between(
                    data['kln'],
                    data['Pln'] - layer * data['dPln'],
                    data['Pln'] + layer * data['dPln'],
                    facecolor=COLOUR['default'],
                    alpha=SHADE_TRANSPARENCY**layer
                )
        elif error_sty == 'bar':
            plt.errorbar(
                data['kln'],
                data['Pln'],
                1 * data['dPln'],
                fmt='o',
                markersize=MARKER_SIZE,
                color=COLOUR['default'],
                capsize=0.,
                elinewidth=ERROR_LINEWIDTH,
                ecolor=COLOUR['default'],
                alpha=BAR_TRANSPARENCY**1,
                label=LABEL['default']
            )
            plt.errorbar(
                data['kln'],
                data['Pln'],
                2 * data['dPln'],
                fmt='o',
                markersize=MARKER_SIZE,
                color=COLOUR['default'],
                capsize=0.,
                elinewidth=ERROR_LINEWIDTH,
                ecolor=COLOUR['default'],
                alpha=BAR_TRANSPARENCY**2
            )

        plt.errorbar(
            data['k'],
            data['Pk'],
            xerr=data['dk'],
            yerr=data['dPk'],
            color=COLOUR['reference'],
            label=LABEL['reference']
        )

        for idx, dbl_indices in enumerate(data['ln']):
            if dbl_indices[0] == 0:
                plt.scatter(
                    data['kln'][idx],
                    data['Pln'][idx],
                    marker=MARKER,
                    color=COLOUR['default']
                    )
                plt.annotate(
                    str(dbl_indices),
                    xy=(data['kln'][idx], data['Pln'][idx]),
                    verticalalignment='bottom',
                    fontsize=ANNO_SIZE
                )

        if smoothed_data is not None:
            plt.loglog(
                smoothed_data['kln'],
                smoothed_data['Pln'],
                color=COLOUR['additional'],
                lw=ERROR_LINEWIDTH,
                zorder=5
            )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.legend()

        plt.subplot2grid(
            ERROR_CANVAS_GRID,
            (ERROR_ROW_SPAN, 0),
            rowspan=ERROR_CANVAS_GRID[0]-ERROR_ROW_SPAN,
            colspan=ERROR_CANVAS_GRID[1]
        )

        cartesian_spline = IUSpline(data['k'], data['Pk'])
        if smoothed_data is None:
            plt.plot(
                data['kln'],
                data['Pln'] / cartesian_spline(data['kln']) - 1,
                color=COLOUR['default']
            )
        else:
            plt.plot(
                smoothed_data['kln'],
                smoothed_data['Pln'] \
                    / cartesian_spline(smoothed_data['kln']) - 1,
                color=COLOUR['default'],
            )

        plt.axhline(y=0., ls='--', lw=1., c='k')
        plt.fill_between(
            (data['kln'].min(), data['kln'].max()),
            [ERROR_PATCH_HT]*2,
            [-ERROR_PATCH_HT]*2,
            color='k',
            alpha=0.2,
            edgecolor='face'
        )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.ylim(-ERROR_PANEL_HT, ERROR_PANEL_HT)
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$\Delta P(k)/P_\textrm{Cart}(k)$')

        plt.subplots_adjust(hspace=0, wspace=0)

    return fig
