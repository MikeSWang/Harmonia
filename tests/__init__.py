"""Unit tests of ``Harmonia``.

"""
import os
import sys


def import_test_package():
    """Add package to Python module path for testing.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))


def display_mathematica_query(message: str):
    """Print strings to be used as Mathematica/WolframAlpha queries.

    """
    print("Query: \n{}\n".format(message))


class NamedFunction:
    """Named functions.

    Parameters
    ----------
    name: Name of the function.
    func: Function.

    Attributes
    ----------
    name: Name of the function.
    func: Function.

    """
    name: str
    func: callable

    def __init__(self, name: str, func: callable):
        self.name = name
        self.func = func

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


import_test_package()
