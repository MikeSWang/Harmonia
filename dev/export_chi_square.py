"""Export bias chi-square sampled from simulations..

"""
import numpy as np
import matplotlib.pyplot as plt

from likelihood_rc import PATHOUT
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    collate_data_files,
    confirm_directory_path,
    harmony,
    overwrite_protection,
)


def aggregate_data(output):
    """Aggregate data into a single result.

    Parameters
    ----------
    output : dict
        Output data to be aggregated.

    Returns
    -------
    dict
        Aggregated output data.

    """
    return {
        var: np.average(vals, axis=0) for var, vals in output.items()
    }


def safe_save(data, path, name):
    """Safely save data by checking overwrite path and protections.

    Raises
    ------
    AssertionError
        If path does not exist.
    AssertionError
        If overwrite permission is denied at the output path.

    """
    assert confirm_directory_path(path)
    assert overwrite_protection(path, name)
    np.save(path + name, data)


def process_data(collate_data=False, load_data=False, save=False):
    """Collate, load and export likelihood outputs.

    Parameters
    ----------
    collate_data, load_data, save : bool, optional
        If `True` (default is `False`), collate, load and/or save
        likelihood data.

    """
    data_outpath = f"{PATHOUT}{SCRIPT_NAME}/"
    collate_path = data_outpath + "collated/"

    global output, data

    if collate_data:
        output, count, _ = collate_data_files(
            f"{data_outpath}{FILE_ROOT}-*{FILE_SUFFIX}*.npy",
            'npy'
        )
        data = aggregate_data(output)

        if save:
            save_str = "".join(_.split("(")[-1].split(")")[:-1])
            safe_save(output, collate_path, f"{FILE_ROOT}-({save_str}).npy")

    if load_data:
        output = np.load(
            f"{collate_path}{FILE_ROOT}-({FILE_SUFFIX}).npy"
        ).item()
        data = aggregate_data(output)

    global disc, index_vector

    disc = DiscreteSpectrum(BOXSIZE/2, 'Dirichlet', KMAX)
    index_vector = SphericalArray\
        .build(disc=disc)\
        .unfold(PIVOT, return_only='index')

    global mode_count_by_degree, chi_square_by_degree

    mode_count_by_degree = np.zeros(len(disc.degrees))
    chi_square_by_degree = np.zeros((len(data['f_nl']), len(disc.degrees)))
    for degree in disc.degrees:
        mode_count_by_degree[degree] = (2*degree + 1) * disc.depths[degree]
        selected_slice = list(
            map(lambda index: index[0] == degree, index_vector)
        )
        chi_square_by_degree[:, degree] = np.sum(
            data['chi_square'][:, selected_slice],
            axis=1
        )

    global chi_sq_mode_rejected

    retained_slice = list(
        map(lambda index: index[0] != 0, index_vector)
    )
    chi_sq_mode_rejected = np.sum(
        data['chi_square'][:, retained_slice],
        axis=1
    )


def view_data(savefig=False):
    """Visualise output data.

    Parameters
    ----------
    savefig : bool, optional
        If `True` (default is `False`), save plotted likelihood figure.

    """
    plt.close('all')
    plt.style.use(harmony)
    plt.plot(
        data['f_nl'],
        chi_sq_mode_rejected
    )
#    plt.figure(figsize=(6, 12))
#    LINE_STYLE = ["-", "--", "-."]
#    for degree in disc.degrees:
#        chi_square_by_degree_per_dof = \
#            chi_square_by_degree[:, degree] # / mode_count_by_degree[degree]
#        plt.plot(
#            data['f_nl'],
#            chi_square_by_degree_per_dof,
#            ls = LINE_STYLE[degree // 5],
#            label=r"$\ell = {}$".format(degree)
#        )
#        plt.annotate(
#            r"$\ell = {}$".format(degree),
#            xy=(data['f_nl'][-1], chi_square_by_degree_per_dof[-1]),
#            horizontalalignment='left',
#            fontsize=10
#        )
#
#    plt.xlim(-200., 250.)
    plt.xlabel(r"$f_\mathrm{NL}$")
    plt.ylabel(r"$\chi^2(f_\mathrm{NL})$")
#    plt.legend(
#        bbox_to_anchor=(0., 1.05),
#        loc=3,
#        ncol=3,
#        borderaxespad=0.
#    )


if __name__ == '__main__':

    SCRIPT_NAME = "simulation_likelihood"
    FILE_ROOT = "halos-(NG=0.,z=1.)"

    FILE_SUFFIX = (
        "prior=[-200.0,200.0],pivot=natural,"
        "nbar=2.49e-4,b1=2.4048,f0=none,kmax=0.04"
    )

    PIVOT = 'natural'
    BOXSIZE = 1000.
    KMAX = 0.04

    process_data(collate_data=False, load_data=True, save=True)
    view_data()
