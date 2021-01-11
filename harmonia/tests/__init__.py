"""
***************************************************************************
Unit testing (:mod:`~harmonia.tests`)
***************************************************************************

Unit testing with `pytest` in Python 3.5 or above.  Any public function
is tested only if: it is not a method of a class; has a yield or return
statement; and do not return a string, a callable or a class instance.
Any class that does not depend on a class from an external package is
tested for their public methods, properties and most of the magic methods.

"""


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
