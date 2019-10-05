"""Predict 2-point function values for fixed-epoch N-body simulations.

"""
import warnings
from functools import partial

import numpy as np
from mpi4py import MPI
from scipy.interpolate import interp1d

from agreement_rc import PATHIN, PATHOUT, params, script_name
from harmonia.algorithms import DiscreteSpectrum, SphericalArray
from harmonia.collections import (
    confirm_directory_path as confirm_dir,
    format_float,
    mpi_compute as mpicomp,
)
from harmonia.reader import coupling_list, twopoint_signal, twopoint_shotnoise

PK_FILE_ROOT = "halos-(NG=0.,z=1.)-Pk-(nbar=2.49e-4,b=2.3415)"


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
        nbar = params.nbar
        bias = params.bias
        rmax = params.boxside / 2
        kmax = params.kmax
    except AttributeError as attr_err:
        raise AttributeError(attr_err)

    global Plin, beta

    k_points, Pk_points = np.loadtxt("".join(
        [PATHIN, script_name, "/", PK_FILE_ROOT, ".txt"]
    )).T

    Plin = interp1d(k_points, Pk_points, assume_sorted=True)
    beta = 0.

    if len(pivots) > 1:
        pivot_tag = "{}".format(pivots).replace("'", "")
    else:
        pivot_tag = "{}".format(params.structure).replace("'", "")

    param_tag = "pivots={},nbar={},bias={},beta={},rmax={},kmax={},".format(
        pivot_tag,
        format_float(nbar, 'sci'),
        format_float(bias, 'decdot'),
        format_float(beta, 'decdot'),
        format_float(rmax, 'intdot'),
        format_float(kmax, 'sci'),
    )
    runtime_info = "-({})".format(param_tag)
    return runtime_info

# -- Runtime parameters -------------------------------------------------------



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# == PROCESSING ===============================================================

# Set up discretisation and indexing.
disc = DiscreteSpectrum(rmax, 'Dirichlet', kmax)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    indx_vec = SphericalArray.build(disc=disc).unfold(struct, retonly='index')

# Compute couplings with parallel processes.
in2co = {
    coupletype: partial(coupling_list, coupletype=coupletype, disc=disc)
    for coupletype in ['ang', 'rad', 'rsd']
    }
couplings = mpicomp(indx_vec, in2co, comm)

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


# == FINALISATION =============================================================

if rank == 0:
    fpathful, fnameroot = f"{PATHOUT}{fdir}{subdir}", fname.split("_")[0]
    confirm_dir(fpathful)

    np.save(
        "".join([
            fpathful, fnameroot, "_couplings",
            f"-(nbar=2.49e-4,b=2.3415,kmax={ff(kmax, 'sci')},struct={struct})",
            ".npy"
        ]),
        couplings
        )
    np.save(
        "".join([
            fpathful, fnameroot, "_2pt",
            f"-(nbar=2.49e-4,b=2.3415,kmax={ff(kmax, 'sci')},struct={struct})",
            ".npy"
        ]),
        {'signal': cov_signal,
         'shotnoise': cov_shotnoise,
         'covar': cov_signal + cov_shotnoise,
         }
        )
