import unittest
import os
import sys
from random import randint
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datareader import DataReader
import word2vec as wv
from util import run_test, get_path_basic_corpus


class Testopt(unittest.TestCase):
    """
    Class that test the basic optmization
    """
    def test_run_training(self):
        """
        Test to check if the read_text function
        return a list of words given a txt file.
        """
        my_data = DataReader(get_path_basic_corpus())
        my_vocab_size = 500
        my_data.process_data(my_vocab_size)
        my_config = wv.Config(num_steps=200,
                              vocab_size=my_vocab_size,
                              show_step=2)

        my_model = wv.SkipGramModel(my_config)
        duration, loss = wv.run_training(my_model,
                                         my_data,
                                         verbose=False,
                                         visualization=False,
                                         debug=True)
        self.assertTrue(duration <= 1.7)
        self.assertTrue(loss < 7)

if __name__ == "__main__":
    run_test(Testopt,
             "\n=== Running opt tests ===\n")
