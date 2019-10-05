"""Visualise recovered power spectra.

"""
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline

ANNO_SIZE = 6
COLOUR = {'reference': '#C40233', 'default': '#0087BD'}
ERROR_CANVAS_GRID = (4, 8)
ERROR_ROW_SPAN = 3
LABEL = {'reference': 'Cartesian', 'default': 'spherical'}
MARKER = '+'
TRANSPARENCY = 1/4


def view_spectrum(data, case='error'):
    """Plot power spectra.

    Parameters
    ----------
    data : dict
        Power spectrum data.
    case : {'single', 'error'}, optional
        Plotting case, either a single plot (``'single'``) or with error
        panel (``'error'``).

    """
    if case == 'single':

        fig = plt.figure()

        plt.errorbar(
            data['k'],
            data['Pk'],
            xerr=data['dk'],
            yerr=data['dPk'],
            color=COLOUR['reference'],
            label=LABEL['reference']
        )

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
                alpha=TRANSPARENCY**layer
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
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.legend()

    if case == 'error':

        fig = plt.figure(figsize=(7, 5.5))

        plt.subplot2grid(
            ERROR_CANVAS_GRID,
            (0, 0),
            rowspan=ERROR_ROW_SPAN,
            colspan=ERROR_CANVAS_GRID[1]
        )

        plt.errorbar(
            data['k'],
            data['Pk'],
            xerr=data['dk'],
            yerr=data['dPk'],
            color=COLOUR['reference'],
            label=LABEL['reference']
        )

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
                alpha=TRANSPARENCY**layer
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
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.legend()

        plt.subplot2grid(
            ERROR_CANVAS_GRID,
            (ERROR_ROW_SPAN, 0),
            rowspan=ERROR_CANVAS_GRID[0]-ERROR_ROW_SPAN,
            colspan=ERROR_CANVAS_GRID[1]
        )

        cartesian_spline = IUSpline(data['k'], data['Pk'])(data['kln'])

        plt.plot(
            data['kln'],
            data['Pln'] / cartesian_spline - 1,
            c=COLOUR['default']
        )
        plt.axhline(y=0., ls='--', lw=1., c='k')

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$\Delta P(k)/P_\textrm{Cart}(k)$')

        plt.subplots_adjust(hspace=0, wspace=0)

    return fig
