import numpy as np
from scipy.interpolate import interp1d

K_SAMP, PK_SAMP = np.loadtxt("./data/input/Pk_dm_z1.txt").T

def Plin(k):
    interp_power_spectrum = interp1d(K_SAMP, PK_SAMP, assume_sorted=True)
    return interp_power_spectrum(k)
