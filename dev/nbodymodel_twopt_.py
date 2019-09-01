import numpy as np
from scipy.interpolate import interp1d

from twoptrc import PATHIN

FILE = "Pk-(NG=0.,z=1.)-(nbar=2.49e-4,b=2.3415)"
k_samp, Pk_samp = np.loadtxt("".join([PATHIN, FILE, ".txt"])).T

Plin = interp1d(k_samp, Pk_samp, assume_sorted=True)
