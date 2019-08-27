"""
Style sheet (:mod:`~concordia.collections.style`)
===============================================================================

:mod:`matplotlib` visual style.

"""
import os.path

import matplotlib

_cwd = os.path.dirname(__file__)
_mplstyle_path = os.path.join(_cwd, 'harmony.mplstyle')

globals()['harmony'] = matplotlib.rc_params_from_file(
    _mplstyle_path, use_default_template=False
    )
