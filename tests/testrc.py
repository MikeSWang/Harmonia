import os
import sys

from pytest import approx

sys.path.insert(
    0,
    os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            "../",
        ),
    ),
)


class NamedFunction:

    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def wolfram_alpha_query(message):
    print("WolframAlpha query: \n{}\n".format(message))
