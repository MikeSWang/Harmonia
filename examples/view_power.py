"""Visualise recovered power spectrum.

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline


def view_spectrum(data, data_add=None, case='single', config=None):
    """Visualise real-space power spectrum.

    Parameters
    ----------
    data : dict
        Power spectrum data to be plotted.
    data_add : dict or None, optional
        Additional power spectrum data to be plotted (default is `None`).
    case : {'single', 'error', 'overlay', 'panel'}, optional
        Plot type, ``'single'`` for simple plots, ``'error'`` for relative
        error bottom panel, ``'overlay'`` for comparison within the same plot,
        ``'panel'`` for side by side panels with relative error bottom panels.
    config : dict or None, optional
        Additional options to be passed on to the figure (default is `None`).

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Power spectrum plot.

    """
    # Override default plotting options wherever applicable.
    CARTESIAN_COLOUR = '#C40233'
    SPHERICAL_COLOUR = '#0087BD'
    SPHERICAL_COLOUR_ADD = '#009F6B'

    ERROR_CANVAS_GRID = (4, 8)
    ERROR_MROWSPAN = 3
    PANEL_CANVAS_GRID = (4, 15)
    PANEL_ROWSPAN, PANEL_WSPACE, PANEL_COLSPAN = 3, 1, 7

    TRANSPARENCY_1, TRANSPARENCY_2 = 1/4, 1/16
    ANNO_SIZE = 7

    options = {
        'close_all': True,
        'cartesian_label': 'Cartesian',
        'spherical_label': 'spherical',
        'cartesian_label_add': 'Cartesian',
        'spherical_label_add': 'spherical',
        'title': 'Case 1',
        'title_add': 'Case 2',
        'errbarwgt': 0.8,
        'mode_marker': 'x',
        }

    if case == 'panel':
        options['figsize'] = (12, 5.5)
    elif case in ['error', 'overlay']:
        options['figsize'] = (7, 5.5)
    else:
        options['figsize'] = None

    if config is not None:
        for key, val in config.items():
            options[key] = val

    # Plotting canvas.
    if options['close_all']:
        plt.close('all')
    if options['figsize']:
        fig = plt.figure(
            'Real-space power spectrum', figsize=options['figsize']
            )
    else:
        fig = plt.figure('Real-space power spectrum')

    # Plotting cases.
    if case == 'single':
        plt.errorbar(
            data['k'], data['Pk'],
            xerr=data['dk']/np.sqrt(data['dof1']),
            yerr=data['dPk']/np.sqrt(data['dof1']),
            elinewidth=options['errbarwgt'], color=CARTESIAN_COLOUR,
            label=options['cartesian_label']
            )

        plt.loglog(
            data['kln'], data['Pln'], color=SPHERICAL_COLOUR,
            label=options['spherical_label']
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - 2*data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + 2*data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data['kln'][idx], data['Pln'][idx],
                    marker=options['mode_marker'], color=SPHERICAL_COLOUR
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
        plt.subplot2grid(
            ERROR_CANVAS_GRID, (0, 0),
            rowspan=ERROR_MROWSPAN, colspan=ERROR_CANVAS_GRID[1]
            )

        plt.errorbar(
            data['k'], data['Pk'],
            xerr=data['dk']/np.sqrt(data['dof1']),
            yerr=data['dPk']/np.sqrt(data['dof1']),
            elinewidth=options['errbarwgt'], color=CARTESIAN_COLOUR,
            label=options['cartesian_label']
            )

        plt.loglog(
            data['kln'], data['Pln'], color=SPHERICAL_COLOUR,
            label=options['spherical_label']
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - 2*data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + 2*data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data['kln'][idx], data['Pln'][idx],
                    marker=options['mode_marker'], color=SPHERICAL_COLOUR
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
            ERROR_CANVAS_GRID, (ERROR_MROWSPAN, 0),
            rowspan=ERROR_CANVAS_GRID[0]-ERROR_MROWSPAN,
            colspan=ERROR_CANVAS_GRID[1]
            )

        cart = IUSpline(data['k'], data['Pk'])(data['kln'])

        plt.plot(data['kln'], (data['Pln']-cart)/cart, '--k')  # modifiable
        plt.axhline(y=0, c='k', alpha=.5)  # modifiable

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$\Delta P(k)/P_\textrm{Cart}(k)$')

    if case == 'overlay':
        if data_add is None:
            raise ValueError("Comparison to be overlayed missing. ")

        plt.subplot2grid(
            ERROR_CANVAS_GRID, (0, 0),
            rowspan=ERROR_MROWSPAN, colspan=ERROR_CANVAS_GRID[1]
            )

        plt.errorbar(
            data['k'], data['Pk'],
            xerr=data['dk']/np.sqrt(data['dof1']),
            yerr=data['dPk']/np.sqrt(data['dof1']),
            elinewidth=options['errbarwgt'], color=CARTESIAN_COLOUR,
            label=options['cartesian_label']
            )

        plt.errorbar(
            data_add['k'], data_add['Pk'],
            xerr=data_add['dk']/np.sqrt(data_add['dof1']),
            yerr=data_add['dPk']/np.sqrt(data_add['dof1']),
            elinewidth=options['errbarwgt'], color=CARTESIAN_COLOUR,
            label=options['cartesian_label_add'],
            ls='--'  # modifiable
            )

        plt.loglog(
            data['kln'], data['Pln'], color=SPHERICAL_COLOUR,
            label=options['spherical_label']
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - 2*data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + 2*data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_2
            )

        plt.loglog(
            data_add['kln'], data_add['Pln'],
            color=SPHERICAL_COLOUR_ADD, label=options['spherical_label_add'],
            ls='--'  # modifiable
            )
        plt.fill_between(
            data_add['kln'],
            data_add['Pln'] - data_add['dPln']/np.sqrt(data_add['dof2']),
            data_add['Pln'] + data_add['dPln']/np.sqrt(data_add['dof2']),
            facecolor=SPHERICAL_COLOUR_ADD, alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data_add['kln'],
            data_add['Pln'] - 2*data_add['dPln']/np.sqrt(data_add['dof2']),
            data_add['Pln'] + 2*data_add['dPln']/np.sqrt(data_add['dof2']),
            facecolor=SPHERICAL_COLOUR_ADD, alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data['kln'][idx], data['Pln'][idx],
                    marker=options['mode_marker'],
                    color=SPHERICAL_COLOUR
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
            ERROR_CANVAS_GRID, (ERROR_MROWSPAN, 0),
            rowspan=ERROR_CANVAS_GRID[0]-ERROR_MROWSPAN,
            colspan=ERROR_CANVAS_GRID[1]
            )

        cart0 = IUSpline(data['k'], data['Pk'])(data['kln'])
        cart1 = IUSpline(data_add['k'], data_add['Pk'])(data_add['kln'])

        # Modifiable!
        plt.plot(data['kln'], (data['Pln']-cart0)/cart0, '-k')
        plt.plot(data_add['kln'], (data_add['Pln']-cart1)/cart1, ':k')
        plt.axhline(y=0, ls=':', c='k', alpha=.5)

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$\Delta P(k)/P_\textrm{Cart}(k)$')

    if case == 'panel':
        if data_add is None:
            raise ValueError("Comparison to be overlayed missing. ")

        g0 = plt.subplot2grid(
            PANEL_CANVAS_GRID, (0, 0),
            rowspan=PANEL_ROWSPAN, colspan=PANEL_COLSPAN
            )

        plt.errorbar(
            data['k'], data['Pk'],
            xerr=data['dk']/np.sqrt(data['dof1']),
            yerr=data['dPk']/np.sqrt(data['dof1']),
            elinewidth=options['errbarwgt'], color=CARTESIAN_COLOUR,
            label=options['cartesian_label']
            )

        plt.loglog(
            data['kln'], data['Pln'], color=SPHERICAL_COLOUR,
            label=options['spherical_label']
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data['kln'],
            data['Pln'] - 2*data['dPln']/np.sqrt(data['dof2']),
            data['Pln'] + 2*data['dPln']/np.sqrt(data['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data['kln'][idx], data['Pln'][idx],
                    marker=options['mode_marker'], color=SPHERICAL_COLOUR
                    )
                plt.annotate(
                    r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                    xy=(data['kln'][idx], data['Pln'][idx]),
                    verticalalignment='bottom', fontsize=ANNO_SIZE
                    )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
        plt.title(options['title'])

        yl0, yh0 = plt.gca().get_ylim()

        g1 = plt.subplot2grid(
            PANEL_CANVAS_GRID, (0, PANEL_COLSPAN+PANEL_WSPACE),
            rowspan=PANEL_ROWSPAN, colspan=PANEL_COLSPAN
            )

        plt.errorbar(
            data_add['k'], data_add['Pk'],
            xerr=data_add['dk']/np.sqrt(data_add['dof1']),
            yerr=data_add['dPk']/np.sqrt(data_add['dof1']),
            elinewidth=options['errbarwgt'], color=CARTESIAN_COLOUR,
            label=options['cartesian_label']
            )

        plt.loglog(
            data_add['kln'], data_add['Pln'], color=SPHERICAL_COLOUR,
            label=options['spherical_label']
            )
        plt.fill_between(
            data_add['kln'],
            data_add['Pln'] - data_add['dPln']/np.sqrt(data_add['dof2']),
            data_add['Pln'] + data_add['dPln']/np.sqrt(data_add['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_1
            )
        plt.fill_between(
            data_add['kln'],
            data_add['Pln'] - 2*data_add['dPln']/np.sqrt(data_add['dof2']),
            data_add['Pln'] + 2*data_add['dPln']/np.sqrt(data_add['dof2']),
            facecolor=SPHERICAL_COLOUR, alpha=TRANSPARENCY_2
            )

        for idx, ind_lab in enumerate(data_add['ln']):
            if ind_lab[0] == 0:
                plt.scatter(
                    data_add['kln'][idx], data_add['Pln'][idx],
                    marker=options['mode_marker'], color=SPHERICAL_COLOUR
                    )
                plt.annotate(
                    r'$({:d},{:d})$'.format(ind_lab[0], ind_lab[1]),
                    xy=(data_add['kln'][idx], data_add['Pln'][idx]),
                    verticalalignment='bottom', fontsize=ANNO_SIZE
                    )

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.gca().yaxis.tick_right()
        plt.legend()
        plt.title(options['title_add'])

        yl1, yh1 = plt.gca().get_ylim()

        yl, yh = 0.975*min(yl0, yl1), 1.025*max(yh0, yh1)
        g0.set_ylim(bottom=yl, top=yh)
        g1.set_ylim(bottom=yl, top=yh)

        e0 = plt.subplot2grid(
            PANEL_CANVAS_GRID, (PANEL_ROWSPAN, 0),
            rowspan=PANEL_CANVAS_GRID[0]-PANEL_ROWSPAN, colspan=PANEL_COLSPAN
            )

        cart0 = IUSpline(data['k'], data['Pk'])(data['kln'])

        # Modifiable!
        plt.plot(data['kln'], (data['Pln']-cart0)/cart0, '--k')
        plt.axhline(y=0, ls=':', c='k', alpha=.5)

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.ylabel(r'$\Delta P(k)/P_\textrm{Cart}(k)$')

        yb0, yt0 = plt.gca().get_ylim()

        e1 = plt.subplot2grid(
            PANEL_CANVAS_GRID, (PANEL_ROWSPAN, PANEL_COLSPAN+PANEL_WSPACE),
            rowspan=PANEL_CANVAS_GRID[0]-PANEL_ROWSPAN, colspan=PANEL_COLSPAN
            )

        cart1 = IUSpline(data_add['k'], data_add['Pk'])(data_add['kln'])

        # Modifiable!
        plt.plot(data_add['kln'], (data_add['Pln']-cart1)/cart1, '--k')
        plt.axhline(y=0, ls=':', c='k', alpha=.5)

        plt.xlim(left=0.99*data['kln'].min(), right=1.01*data['kln'].max())
        plt.xscale('log')
        plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
        plt.gca().yaxis.tick_right()

        yb1, yt1 = plt.gca().get_ylim()

        yb, yt = 0.975*min(yb0, yb1), 1.025*max(yt0, yt1)
        e0.set_ylim(bottom=yb, top=yt)
        e1.set_ylim(bottom=yb, top=yt)

    plt.subplots_adjust(hspace=0, wspace=0)

    return fig
