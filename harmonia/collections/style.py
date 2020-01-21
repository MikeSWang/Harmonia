"""
Visualisation style (:mod:`~harmonia.collections.style`)
===========================================================================

:mod:`matplotlib` style sheet.

"""
import os.path

import matplotlib

_mplstyle_dir = os.path.dirname(__file__)
_mplstyle_path = os.path.join(_mplstyle_dir, 'harmony.mplstyle')

harmony = matplotlib.rc_params_from_file(
    _mplstyle_path, use_default_template=False
)
