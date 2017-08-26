import unittest
import os
import sys
from random import randint
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datareader import DataReader
from util import get_time, get_path_basic_corpus, run_test


class TestReading(unittest.TestCase):
    """
    Class that test the reading function
    """
    def test_read_text(self):
        """
        Test to check if the read_text function
        return a list of words given a txt file.
        """
        dr1 = DataReader()
        dr2 = DataReader(punctuation=True)
        words1 = dr1.read_text()
        words2 = dr2.read_text()
        print("\nReading time = {}\n".format(get_time(dr1.read_text)))

        self.assertTrue(len(words1) > 0)
        self.assertTrue(len(words2) > 0)
        self.assertEqual(words1[22], "system")
        self.assertEqual(words2[22], "system.")


class TestData(unittest.TestCase):
    """
    Class that test the build_vocab and get_data functions
    """
    @classmethod
    def setUpClass(cls):
        cls.dr = DataReader()
        cls.words = cls.dr.read_text()

    def test_build_vocab(self):
        """
        Test to check if the read_text function
        return a list of words given a txt file.
        """
        vocab_size = 500
        _, dic, revert_dic = self.dr.build_vocab(self.words, vocab_size)
        print("\nBuilding vocab time = {}\n".format(get_time(self.dr.build_vocab,
                                                             vocab_size)))
        self.assertTrue(len(dic) == vocab_size)
        self.assertTrue(len(revert_dic) == vocab_size)

    def test_batch_generator(self):
        """
        Test to check if the batch_generator function chooses a context word
        in the skip_window for each center word
        """
        vocab_size = 4208
        self.dr.process_data(vocab_size)
        data_index = 0
        skip_window = randint(1, 50)
        num_skips = max(int(skip_window/2), 2)
        batch_size = num_skips*3
        new_index, batch, label = self.dr.batch_generator(batch_size,
                                                          num_skips,
                                                          skip_window,
                                                          data_index)
        batch = list(batch)
        for i, word in enumerate(self.dr.data[0:new_index]):
            while word in batch and skip_window <= i:
                index = batch.index(word)
                context = label[index][0]
                before = self.dr.data[i-skip_window:i]
                after = self.dr.data[i+1:i+skip_window+1]
                self.assertTrue(context in before or context in after)
                batch[index] = -1
        print("\nBuilding bacth time = {}".format(get_time(self.dr.batch_generator,
                                                           [batch_size,
                                                            data_index])))


if __name__ == "__main__":
    run_test(TestReading,
             "\n=== Running reading tests ===\n")
    run_test(TestData,
             "\n=== Running data tests ===\n")
