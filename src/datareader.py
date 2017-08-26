import util
import string
from collections import Counter, deque
import os
import numpy as np
from random import randint


class DataReader(object):
    """
    Class to read and manipulate text.
    """
    def __init__(self,
                 path=None,
                 punctuation=False,
                 write_vocab=True):
        """
        :type path: string >>> path to text
        :type punction: boolean
        :type write_vocab: boolean
        """
        if not path:
            path = util.get_path_basic_corpus()

        self.path = path
        self.punctuation = punctuation
        self.write_vocab = write_vocab

    @util.timeit()
    def read_text(self):
        """
        Given a path to a txt file 'path' this function
        reads each line of the file and stores each word in a list words.
        'punctuation' is a parameter to control if we want the punctuation
        of the text to be captured by this reading or not.

        :type path: string
        :type punctuation: boolean
        :rtype: list of strings
        """

        dic_trans = {key: None for key in string.punctuation}
        translator = str.maketrans(dic_trans)
        words = []
        with open(self.path) as inputfile:
            for line in inputfile:
                line = line.lower()
                if not self.punctuation:
                    line = line.translate(translator)

                words.extend(line.strip().split())
        return words

    @util.timeit([2])
    def build_vocab(self, words, vocab_size):
        """
        Given one list of words 'words' and
        one int 'vocab_size' this functions constructs
        one list of (word, frequency) named 'count' of size vocab_size
        (only the vocab_size - 1 most frequent words are here, the rest will
        be discarded as 'UNK'). This function returns also two dicts
        'word2index' and 'index_to_word' to translate the words in
        indexes and vice-versa.
        The parameter 'write_vocab' controls if you want to creat a file
        'vocab_1000.tsv' for vector vizualization in Tensorboard.

        :type words: list of strings
        :type vocab_size: int
        :type write: boolean
        :rtype count: list of tuples -> (str,int)
        :rtype word2index: dictionary
        :rtype index2word: dictionary
        """
        count = [("UNK", 0)]
        most_frequent_words = Counter(words).most_common(vocab_size - 1)
        count.extend(most_frequent_words)
        word2index = {}
        index = 0

        if self.write_vocab:
            path = os.path.dirname(__file__)
            path = os.path.join(path, 'vocab_1000.tsv')
            f = open(path, "w")

        for word, _ in count:
            word2index[word] = index

            if index < 1000 and self.write_vocab:
                f.write(word + "\n")

            index += 1

        if self.write_vocab:
            f.close()

        index2word = dict(zip(word2index.values(), word2index.keys()))
        return count, word2index, index2word

    @util.timeit([1])
    def process_data(self, vocab_size=50000):
        """
        This function transform the text "words" into a list
        of numbers according to the dictionary word2index.
        It also modifies the frequency counter 'count' to
        count the frequency of the word 'UNK'.

        :type words: list of strings
        :type count: list of tuples -> (str,int)
        :type word2index: dictionary
        :type index_to_word: list of dictionary
        :rtype data: list of ints
        :rtype count: list of tuples -> (str,int)

        :rtype data: list of ints
        :rtype count: list of tuples -> (str,int)
        :rtype word2index: dictionary
        :rtype index2word: list of dictionary
        """
        words = self.read_text()
        self.count, self.word2index, self.index2word = self.build_vocab(words,
                                                                        vocab_size)
        self.data = []
        unk_count = 0
        for word in words:
            index = self.word2index.get(word, 0)

            if not index:
                unk_count += 1

            self.data.append(index)

        self.count[0] = ('UNK', unk_count)

    @util.timeit([1, 4])
    def batch_generator(self,
                        batch_size,
                        num_skips,
                        skip_window,
                        data_index):

        """
        This functions goes thought the processed text 'data' (starting at
        the point 'data_index') and at each step creates a reading window
        of size 2 * skip_window + 1. The word in the center of this
        window will be the center word and it is stored in the array
        'batch'; this function also chooses at random one of the remaining
        words of the window and store it in the array 'labels'. The
        parameter num_skips controls how many times we will use the same center
        word. After all this processing the point in the text has changed, so
        this function also return the number 'data_index'.



        :type batch_size: int
        :type num_skips: int
        :type skip_window: int
        :type data_index: int
        :type data: list of ints
        :rtype data_index: int
        :rtype batch: np array -> [shape = (batch_size), dtype=np.int32]
        :rtype labels: np array -> [shape = (batch_size,1), dtype=np.int32]
        """
        if batch_size % num_skips != 0:
            raise ValueError(
                """batch_size ({0}) should be a multiple of num_skips ({1})""".format(batch_size, num_skips))
        if num_skips > 2 * skip_window:
            raise ValueError(
                """num_skips ({0}) should be less or equal than twice
                the value of skip_window ({1})""".format(num_skips, skip_window))

        data_size = len(self.data)
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        reading_window = deque(maxlen=span)
        for _ in range(span):
            reading_window.append(self.data[data_index])
            data_index = (data_index + 1) % data_size
        for i in range(int(batch_size / num_skips)):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = randint(0, span - 1)
                targets_to_avoid.append(target)
                center_word = reading_window[skip_window]
                context_word = reading_window[target]
                batch[i * num_skips + j] = center_word
                labels[i * num_skips + j, 0] = context_word
            reading_window.append(self.data[data_index])
            data_index = (data_index + 1) % data_size

        return data_index, batch, labels
