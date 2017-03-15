import time
import os
import inspect

timing = {}


def get_time(f, args):
    """
    After using timeit we can get the duration of the function f
    when it was applied in parameters args. Normally it is expected
    that args is a list of parameters, but it can be also a single parameter.

    :type f: function
    :type args: list
    :rtype: float
    """
    if type(args) != list:
        args = [args]
    key = f.__name__ + "-" + "_".join([str(arg) for arg in args])
    return timing[key]


def timeit(method):
    """
    Decorator for time information
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        timed.__name__ = method.__name__
        te = time.time()
        fkey = method.__name__ + "-" + "_".join([str(arg) for arg in args])
        timing[fkey] = te-ts
        return result

    return timed


def get_path_basic_corpus():
    """
    Function to acces the path of the testing corpus.

    :rtype: string
    """
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    filepath = os.path.join(currentdir, "data")
    filepath = os.path.join(filepath, "basic_pt.txt")
    return filepath
