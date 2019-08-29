import os
import sys


def wolfram_alpha_query(message):
    print("WolframAlpha query: \n{}\n".format(message))


class NamedFunction:

    def __init__(self, name, func):
        self.func = func
        self.name = name

    def __repr__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


sys.path.insert(
    0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
    )
