"""Predict 2-point function values from a set of cosmological parameters.

"""
import warnings
from functools import partial

import numpy as np
from mpi4py import MPI
from nbodykit.lab import cosmology

from agreement_rc import PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    mpi_compute as mpicomp,
    unit_const,
    zero_const,
)
from harmonia.reader import Couplings, twopoint_signal, twopoint_shotnoise


def initialise():
    """Initialise from input parameters, set up cosmology and return runtime
    information.

    Returns
    -------
    runtime_info : str
        Runtime information.

    Raises
    ------
    AttributeError
        If a required input arameter is missing.

    """
    global pivots, rsd_flag, nbar, bias, redshift, zmax, kmax

    try:
        pivots = params.structure.split(",")
        rsd_flag = params.rsd

        nbar = params.nbar
        bias = params.bias
        redshift = params.redshift
        zmax = params.zmax
        kmax = params.kmax
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin, rmax, beta

    cosmo = cosmology.Planck15
    Plin = cosmo.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
    rmax = cosmo.comoving_distance(zmax)
    beta = rsd_flag * cosmo.scale_independent_growth_rate(redshift) / bias

    if len(pivots) > 1:
        pivot_tag = "{}".format(pivots).replace("'", "")
    else:
        pivot_tag = "{}".format(params.structure).replace("'", "")

    if rsd_flag:
        rsd_tag = "{:.2f}".format(beta)
    else:
        rsd_tag = 'none'

    param_tag = "pivots={},nbar={},bias={},beta={},rmax={},kmax={},".format(
        pivot_tag,
        format_float(nbar, 'sci'),
        format_float(bias, 'decdot'),
        rsd_tag,
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
    )
    runtime_info = "-({})".format(param_tag)
    return runtime_info


def process(runtime_info):
    """Program process.

    Parameters
    ----------
    runtime_info : str
        Program runtime information.

    Returns
    -------
    output_data : dict
        Program output.

    """
    print(runtime_info.strip("-"))

    disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
    couplings = Couplings(disc=disc)

    triplet_index_vector = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for pivot in pivots:
            triplet_index_vector[pivot] = SphericalArray.build(disc=disc)\
                .unfold(pivot, return_only='index')

    compiling_maps = {
        coupling_type: lambda mu: couplings.compile_over_index(
            mu,
            coupling_type=coupling_type,
        )
        for coupling_type in ['ang', 'rad', 'rsd']
    }
    coupling_coefficients = {
        {
            coupling_type: mpicomp(
                triplet_index_vector[pivot],
                compiling_maps[coupling_type],
                comm,
            )
            for coupling_type in ['ang', 'rad', 'rsd']
        }
        for pivot in pivots
    }

    # Compute couplings with parallel processes.
    in2co = {
        coupletype: partial(coupling_list, coupletype=coupletype, disc=disc)

        }
    couplings =

    # Compute 2-pt values.
    if rank == 0:
        cov_signal = np.zeros((disc.nmode, disc.nmode), dtype=complex)
        cov_shotnoise = np.zeros((disc.nmode, disc.nmode), dtype=complex)
        for rowidx in range(disc.nmode):
            for colidx in range(rowidx+1):
                cov_signal[rowidx, colidx] = bias**2 * twopoint_signal(
                    Plin, beta, disc,
                    M_mu_all=couplings['ang'][rowidx],
                    M_nu_all=couplings['ang'][colidx],
                    Phi_mu_all=couplings['rad'][rowidx],
                    Phi_nu_all=couplings['rad'][colidx],
                    Upsilon_mu_all=couplings['rsd'][rowidx],
                    Upsilon_nu_all=couplings['rsd'][colidx]
                    )

                mu, nu = indx_vec[rowidx], indx_vec[colidx]
                M_munu = couplings['ang'][rowidx][nu[0]][nu[1]+nu[0]]
                cov_shotnoise[rowidx, colidx] = twopoint_shotnoise(
                    mu, nu, nbar, disc, M_munu
                    )

        idx_upper = np.triu_indices(disc.nmode, k=1)
        cov_signal[idx_upper] = cov_signal.T[idx_upper]
        cov_shotnoise[idx_upper] = cov_shotnoise.T[idx_upper]

    return output_data


def finalise(output_data, save=True):
    """Program finalisation with optional data and figure saving.

    Parameters
    ----------
    output_data : dict
        Program output.
    save : bool, optional
        If `True`, aggregate data over all iterations is saved as a dictionary.

    Raises
    ------
    AssertionError
        If the output path does not exist.

    """
    base_path = f"{PATHOUT}{script_name}"
    assert confirm_dir(base_path)

    filename = f"{script_name}{program_tag}"
    if save:
        np.save("".join([base_path, "/", filename, ".npy"]), output_data)

# -- Cosmology ----------------------------------------------------------------

# -- Program identifier -------------------------------------------------------

if rsd:
    subdir, rsd_tag = "wrsd/", "{:.2f}".format(beta)
else:
    subdir, rsd_tag = "nrsd/", 'none'

ftag = (
    f"-("
    f"nbar={ff(nbar, 'sci')},b={ff(bias, 'decdot')},rsd={rsd_tag},"
    f"rmax={ff(rmax, 'intdot')},kmax={ff(kmax, 'sci')},ord={struct}"
    f")"
    )


# == PROCESSING ===============================================================

if rank == 0: print(ftag)

# Set up discretisation and indexing.
disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)


# == FINALISATION =============================================================

if rank == 0:
    fpathful, fnameroot = f"{PATHOUT}{fdir}{subdir}", fname.split("_")[0]
    confirm_dir(fpathful)

    np.save(
        "".join([fpathful, fnameroot, "_couplings", ftag, ".npy"]),
        couplings
        )
    np.save(
        "".join([fpathful, fnameroot, "_2pt", ftag, ".npy"]),
        {'signal': cov_signal,
         'shotnoise': cov_shotnoise,
         'covar': cov_signal + cov_shotnoise,
         }
        )



if __name__ == '__main__':

    SURVEY_SPECS = {
        'mask': unit_const,
        'selection': unit_const,
        'weight': unit_const,
        'weight_derivative': zero_const
    }

    comm = MPI.COMM_WORLD

    program_tag = initialise()
    output_data = process(program_tag)
    finalise(output_data)
