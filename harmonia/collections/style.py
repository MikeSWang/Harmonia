"""
Style sheet (:mod:`~concordia.collections.style`)
===============================================================================

:mod:`matplotlib` visual style.

"""
import os.path

import matplotlib

_cwd = os.path.split(os.path.abspath(__file__))[0]

globals()['harmony'] = matplotlib.rc_params_from_file(
    os.path.join(_cwd, 'harmony.mplstyle'), use_default_template=False
    )
