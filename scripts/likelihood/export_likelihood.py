"""Export sampled likelihood.

"""
import matplotlib.pyplot as plt
import numpy as np

from likelihood_rc import PATHOUT
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    collate_data_files,
    confirm_directory_path,
    harmony,
    overwrite_protection
)


def safe_save(data, path, file):
    """Safely save data by checking overwrite protections.

    Parameters
    ----------
    data : array_like
        Data to be saved.
    path, file : str or :class:`pathlib.Path`
        Path and file name for the data to be saved.

    """
    if overwrite_protection(path, file):
        try:
            np.save(path/file, data)
        except TypeError:
            np.save(path + file, data)


def filter_data(full_data, remove_degrees=()):
    """Filter data.

    Parameters
    ----------
    full_data : dict
        Collated raw data.
    remove_degrees : int, array_like, optional
        If not an empty tuple (default), modes whose spherical degree is an
        element are removed from the data vector and parametrised
        covariance.

    Returns
    -------
    filtered_data : dict
        Filtered data.

    """
    BOXSIZE = 1000.
    PIVOT = 'spectral'

    disc = DiscreteSpectrum(BOXSIZE/2, 'dirichlet', KHYB)

    index_vector = SphericalArray\
        .build(disc=disc)\
        .unfold(PIVOT, return_only='index')

    excluded_deg = np.fromiter(
        map(lambda index: index[0] in remove_degrees, index_vector),
        dtype=bool
    )

    likelihood_contributions = \
        full_data['spherical_likelihood'][..., ~excluded_deg]

    filtered_data = full_data.copy()
    filtered_data.update(
        {'spherical_likelihood': np.sum(likelihood_contributions, axis=-1)}
    )

    return filtered_data


def read_data(collate_data=False, load_data=False, save_data=False,
              remove_degs=()):
    """Collate, load and export likelihood outputs.

    Parameters
    ----------
    collate_data, load_data, save_data : bool, optional
        If `True` (default is `False`), collate, load or save data.
    remove_degs : int, array_like, optional
        If not an empty tuple (default), modes whose spherical degree is an
        element are removed from the spherical likelihood.

    Returns
    -------
    collated_output : dict
        Processed output data.

    """
    scr_outpath = PATHOUT/script_name
    collation_outpath = scr_outpath/"collated"

    if MAP == "hybrid":
        search_root = f"map={MAP},*knots=[{KHYB},{KMAX}],*{PRIOR}{FIXED}"\
            .replace("=[", "=[[]").replace("],", "[]],")
        program_root = f"map={MAP},knots=[{KHYB},{KMAX}],{PRIOR}{FIXED}"
    else:
        search_root = f"map={MAP},*kmax={KMAX},*{PRIOR}{FIXED}"\
            .replace("=[", "=[[]").replace("],", "[]],")
        program_root = f"map={MAP},kmax={KMAX},{PRIOR}{FIXED}"

    file_name = f"{script_name}-{file_root}-({program_root})"
    if collate_data:
        collated_output, _, _ = collate_data_files(
            f"{scr_outpath}/*{file_root}-*-*{search_root}*.npy", 'npy'
        )

        if FIXED:
            collated_output['parameter'] = x_parameters
        else:
            collated_output['parameter_x'] = x_parameters
            collated_output['parameter_y'] = y_parameters

        if MAP == "hybrid" or MAP == "spherical":
            filtered_output = filter_data(
                collated_output, remove_degrees=remove_degs
            )
        else:
            filtered_output = collated_output.copy()

        if MAP == "hybrid":
            collated_output['likelihood'] = np.squeeze(
                filtered_output['spherical_likelihood']
                + filtered_output['cartesian_likelihood']
            )
            del collated_output['spherical_likelihood']
            del collated_output['cartesian_likelihood']
        else:
            collated_output['likelihood'] = np.squeeze(
                filtered_output[f'{MAP}_likelihood']
            )
            del collated_output[f'{MAP}_likelihood']

        if save_data:
            confirm_directory_path(collation_outpath)
            safe_save(collated_output, collation_outpath, file_name + ".npy")

    if load_data:
        collated_output = np.load(
            collation_outpath/str(file_name + ".npy")
        ).item()

    return collated_output


def view_data(data_to_view, savefig=False, **plot_kwargs):
    """Visualise output data.

    Parameters
    ----------
    data_to_view : (list of) dict
        Data set(s) to visualise.
    savefig : bool, optional
        If `True` (default is `False`), save plotted likelihood figure.
    **plot_kwargs
        Keyword arguments to be passed to the plotting routine.

    """
    from view_likelihood import view_pdf, view_contour

    plt.close('all')
    plt.style.use(harmony)

    if FIXED:
        view_pdf(
            data_to_view,
            r"$f_{\mathrm{NL}}$", r"$\mathcal{L}(f_{\mathrm{NL}})$",
            **plot_kwargs
        )
    else:
        view_contour(
            data_to_view,
            r"$f_{\mathrm{NL}}$", r"$b_1$",
            **plot_kwargs
        )

    if savefig:
        if MAP == "hybrid":
            program_root = f"map={MAP},knots=[{KHYB},{KMAX}],{PRIOR}{FIXED}"
        else:
            program_root = f"map={MAP},kmax={KMAX},{PRIOR}{FIXED}"
        file_name = f"{script_name}-{file_root}-({program_root}).pdf"
        plt.savefig(PATHOUT/script_name/file_name)


if __name__ == '__main__':

    # NOTE: Change this before running.
    MAP = "hybrid"
    NG = 100
    KHYB = 0.04
    KMAX = 0.1
    PRIOR = "bias_prior=[2.15,2.55],fnl_prior=[-100.0,250.0]"
    FIXED = ""

    # NOTE: Change this before running.
    script_name = f"{MAP}_likelihood"
    file_root = f"(NG={int(NG)}.,z=1.)"

    x_parameters = np.linspace(-100.0, 250.0, 351)
    y_parameters = np.linspace(2.15, 2.55, 41)

    # NOTE: Change this before running.
    output = read_data(
        collate_data=False,
        load_data=True,
        save_data=False
    )
    view_data(
        output,
        precision=(0, 2),
        estimate='median',
        truth=(NG, None),
        plot_ranges=([-100.0, 250.0], [2.25, 2.55]),
        # scatter_plot=True,
        # savefig=True,
    )
