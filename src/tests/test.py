import unittest
import os
import sys
import inspect
from random import randint
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_process import read_text, build_vocab, get_data, batch_generator
from util import get_time, get_path_basic_corpus


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.
    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestReading(unittest.TestCase):
    """
    Class that test the reading function
    """
    def test_rt(self):
        """
        Test to check if the read_text function
        return a list of words given a txt file.
        """
        filepath = get_path_basic_corpus()
        words1 = read_text(filepath)
        words2 = read_text(filepath, True)
        print("\nReading time = {}".format(get_time(read_text,
                                                    filepath)))

        self.assertTrue(len(words1) > 0)
        self.assertTrue(len(words2) > 0)
        self.assertTrue(words1[22] == "System")
        self.assertTrue(words2[22] == "System.")


class TestData(unittest.TestCase):
    """
    Class that test the build_vocab and get_data functions
    """
    def test_bv(self):
        """
        Test to check if the read_text function
        return a list of words given a txt file.
        """
        words = read_text(get_path_basic_corpus())
        vocab_size = 500
        _, dic, revert_dic = build_vocab(words, vocab_size, False)
        print("\nBuilding vocab time = {}".format(get_time(build_vocab,
                                                           [words,
                                                            vocab_size,
                                                            False])))
        self.assertTrue(len(dic) == vocab_size)
        self.assertTrue(len(revert_dic) == vocab_size)

    def test_gd(self):
        """
        Test to check if the get_data function
        return a list with the right indexes
        of words comparing to the list words
        """
        words = read_text(get_path_basic_corpus())
        vocab_size = 4208
        count, dic, revert_dic = build_vocab(words, vocab_size, False)
        data, _ = get_data(words, count, dic)
        text_list1 = words
        text_list2 = [revert_dic[index] for index in data]
        comparison = [w1 == w2 for (w1, w2) in zip(text_list1, text_list2)]
        print("\nBuilding data time = {}".format(get_time(get_data,
                                                          [words,
                                                           count,
                                                           dic])))
        self.assertTrue(all(comparison))

    def test_bg(self):
        """
        Test to check if the batch_generator function chooses a context word
        in the skip_window for each center word
        """
        words = read_text(get_path_basic_corpus())
        vocab_size = 4208
        count, dic, revert_dic = build_vocab(words, vocab_size, False)
        data, _ = get_data(words, count, dic)
        data_index = 0
        skip_window = randint(1, 50)
        num_skips = max(int(skip_window/2), 2)
        batch_size = num_skips*3
        new_index, batch, label = batch_generator(batch_size,
                                                  num_skips,
                                                  skip_window,
                                                  data_index,
                                                  data)
        batch = list(batch)
        for i, word in enumerate(data[0:new_index]):
            while word in batch and skip_window <= i:
                index = batch.index(word)
                context = label[index][0]
                before = data[i-skip_window:i]
                after = data[i+1:i+skip_window+1]
                self.assertTrue(context in before or context in after)
                batch[index] = -1
        print("\nBuilding bacth time = {}".format(get_time(batch_generator,
                                                           [batch_size,
                                                            num_skips,
                                                            skip_window,
                                                            data_index,
                                                            data])))


if __name__ == "__main__":
    run_test(TestReading,
             "\n=== Running reading tests ===\n")
    run_test(TestData,
             "\n=== Running data tests ===\n")
