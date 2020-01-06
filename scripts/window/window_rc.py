"""Runtime configuration for likelihood sampling.

"""
import os
import sys
import warnings
from pathlib import Path

PATHIN = Path("../../data/survey/")
PATHOUT = Path("../../data/window/")


def import_local_package():
    """Add package to Python module path.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../../")))


if __name__ != '__main__':

    import_local_package()

    from harmonia.collections import clean_warning_format, get_filename

    warnings.formatwarning = clean_warning_format

    script_name = get_filename(sys.argv[0])
