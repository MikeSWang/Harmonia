"""Unit tests of ``Harmonia``.

"""
import os
import sys


def import_test_package():
    """Add package to Python module path for testing.

    """
    _cwd = os.path.dirname(__file__)
    sys.path.insert(0, os.path.realpath(os.path.join(_cwd, "../")))


def mathematica_query(message):
    """Print our strings to be used as Mathematica/WolframAlpha queries.

    """
    print("Mathematica query: \n{}\n".format(message))


class NamedFunction:
    """Named functions.

    Parameters
    ----------
    name : str
        Name of the function.
    func : callable
        Function.

    Attributes
    ----------
    name : str
        Name of the function.
    func : callable
        Function.

    """

    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


if not __name__ == '__main__':
    import_test_package()
