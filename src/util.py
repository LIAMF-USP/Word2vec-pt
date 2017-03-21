import time
import os
import unittest

timing = {}


def get_time(f, args=[]):
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
    key = f.__name__
    if args != []:
        key += "-" + "-".join([str(arg) for arg in args])
    return timing[key]


def timeit(index_args=[]):

    def dec(method):
        """
        Decorator for time information
        """

        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            timed.__name__ = method.__name__
            te = time.time()
            fkey = method.__name__
            for i, arg in enumerate(args):
                if i in index_args:
                    fkey += "-" + str(arg)
            timing[fkey] = te-ts
            return result
        return timed
    return dec


def get_path_basic_corpus():
    """
    Function to acces the path of the testing corpus.

    :rtype: string
    """
    currentdir = os.path.dirname(__file__)
    filepath = os.path.join(currentdir, "data")
    filepath = os.path.join(filepath, "basic_pt.txt")
    return filepath


def newlogname():
    log_basedir = './graphs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')  # e.g. 12-11-2016_18-20-45
    return os.path.join(log_basedir, run_label)


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.
    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)
