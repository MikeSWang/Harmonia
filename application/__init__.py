"""
###########################################################################
Application Pipeline | ``Harmonia``
###########################################################################

This is the application pipeline of ``harmonia``, a Python package
combining spherical and Cartesian map likelihood analyses for large-scale
galaxy clustering.

"""
import logging
import os
import sys
import warnings
from functools import wraps
from pathlib import Path
from pprint import pformat

import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning


def display_args(logger=None, comm=None):
    """Display parsed command-line arguments parsed by a function (
    e.g. `initialise` as a dictionary.

    Parameters
    ----------
    logger : :class:`logging.Logger` *or None, optional*
        If not `None` (default), print to logger.
    comm : :class:`mpi4py.Comm` *or None*, optional
        MPI communicator.

    """
    def decorator(argparse_func):
        @wraps(argparse_func)
        def wrapper(*args, **kwargs):
            parsed_args = argparse_func(*args, **kwargs)

            if comm is None or comm.rank == 0:
                if logger is not None:
                    logger.info(
                        "\n%s\n%s",
                        "---Program parameters---",
                        pformat(vars(parsed_args))
                        .replace("{", " ").replace("}", " ")
                    )
                else:
                    print(
                        "---Program parameters---",
                        pformat(vars(parsed_args))
                        .replace("{", " ").replace("}", " "),
                        "",
                        sep="\n"
                    )

            return parsed_args
        return wrapper
    return decorator


def confirm_directory(dir_path):
    """Ensure a given directoy path exists.

    Parameters
    ----------
    dir_path : str or :class:`pathlib.Path`
        Directory path.

    Returns
    -------
    bool
        `True` if `dir_path` exists or has been created.

    """
    dir_path = Path(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return os.path.exists(dir_path)


def overwrite_protection(output_path):
    """Inspect and modify overwrite permission.

    The function may ask for permission from the user to overwrite the
    file output path at runtime.

    Parameters
    ----------
    output_path : str or :class:`pathlib.Path`
        Output file path.

    Returns
    -------
    bool
        Write permission.

    """
    if os.path.exists(output_path):

        grant_permission = input(
            "Saving would overwrite existing file at destination. "
            "Do you want to continue? (y/[n]) "
        )

        if grant_permission.lower().startswith('y'):
            return True
        return False

    return True


# Application pipeline and data storage directories.
application_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = Path(application_dir).parent/"storage"

# Harmonia package import.
try:
    from harmonia.utils import setup_logger
except ImportError:
    sys.path.insert(0, os.path.join(application_dir, "../"))
    from harmonia.utils import setup_logger

# Application pipeline logger and style sheet.
harmony = matplotlib.rc_params_from_file(
    os.path.join(application_dir, 'harmony.mplstyle'),
    use_default_template=False
)

application_logger = setup_logger()

# Set logging, warning and plotting behaviour.
logging.captureWarnings(True)
application_logger.setLevel(logging.INFO)

warnings.filterwarnings(action='ignore', category=MatplotlibDeprecationWarning)

pyplot.style.use(harmony)
