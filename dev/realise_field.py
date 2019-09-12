import warnings

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology as cosmo

from fieldrc import params
from harmonia.algorithms.fields import (
    generate_gaussian_random_field,
    generate_lognormal_random_field,
    poisson_sample,
    _cal_isotropic_power_spectrum as cal_power
    )
from harmonia.collections import harmony


# == INITIALISATION ===========================================================

# -- Parameters ---------------------------------------------------------------

stat = params.stat
samp = params.smp
nbar = params.nbar
bias = params.bias
redshift = params.redshift
kmax = params.kmax
boxsize = params.boxsize
nmesh = params.nmeshgen


# -- Cosmology ----------------------------------------------------------------

cosmology = cosmo.Planck15
Plin = cosmo.LinearPower(cosmology, redshift=redshift, transfer='CLASS')
mechanism = {
    'Gaussian': generate_gaussian_random_field,
    'lognormal': generate_lognormal_random_field
    }


# == PROCESSING ===============================================================

field_seed, sampling_seed, _ = np.random.RandomState().\
    randint(low=0, high=4294967295, size=3)

# -- Generation -----------------------------------------------------------

field = mechanism[stat](boxsize, nmesh, Plin, bias=bias, seed=field_seed)
if samp:
    field = poisson_sample(field, nbar, boxsize, seed=sampling_seed)


# -- Validation -----------------------------------------------------------

k, Pk, Nk = cal_power(field, boxsize, kmax=kmax)

pk = bias**2 * Plin(k)
if not samp:
    samp_tag, sn_tag = 'unsampled', 'without'
else:
    pk += 1 / nbar
    samp_tag, sn_tag = 'sampled', 'with'

crt_tag = ''
ratio = np.average(Pk/pk)
if not np.isclose(ratio, 1, atol=1e-2):
    pk *= ratio
    crt_tag += r'${:.2f} \times$ '.format(ratio)
    warnings.warn(
        "Power spectrum model upscaled by {:.2f}. ".format(ratio),
        RuntimeWarning
        )

plt.style.use(harmony)
plt.close('all')
plt.figure(f'{samp_tag} biased {stat} random field')

plt.loglog(k, pk, '--', label=f'power spectrum {sn_tag} shot noise')
plt.errorbar(
    k, Pk, yerr=np.sqrt(2/Nk)*Pk, label=f'{crt_tag}{stat} realisation'
    )

plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(r'$P(k)$ [$(\textrm{Mpc}/h)^3$]')
plt.legend()
