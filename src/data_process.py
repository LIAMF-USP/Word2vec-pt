from util import timeit
import string
from collections import Counter, deque
import inspect
import os
import numpy as np
from random import randint


@timeit
def read_text(path, punctuation=False):
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
    with open(path) as inputfile:
        for line in inputfile:
            if not punctuation:
                aux_line = line.translate(translator)
            else:
                aux_line = line
            words.extend(aux_line.strip().split())
    return words


@timeit
def build_vocab(words, vocab_size, write_vocab=True):
    """
    Given one list of words 'words' and
    one int 'vocab_size' this functions constructs
    one list of (word, frequency) named 'count' of size vocab_size
    (only the vocab_size - 1 most frequent words are here, the rest will
    be discarded as 'UNK'). This function returns also two dicts
    'word_to_index' and 'index_to_word' to translate the words in
    indexes and vice-versa.
    The parameter 'write_vocab' controls if you want to creat a file
    'vocab_1000.tsv' for vector vizualization in Tensorboard.

    :type words: list of strings
    :type vocab_size: int
    :type write: boolean
    :rtype count: list of tuples -> (str,int)
    :rtype word_to_index: dictionary
    :rtype index_to_word: dictionary
    """
    count = [("UNK", 0)]
    most_frequent_words = Counter(words).most_common(vocab_size - 1)
    count.extend(most_frequent_words)
    word_to_index = {}
    index = 0
    if write_vocab:
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        path = os.path.join(path, 'vocab_1000.tsv')
        f = open(path, "w")
    for word, _ in count:
        word_to_index[word] = index
        if index < 1000 and write_vocab:
            f.write(word + "\n")
        index += 1
    if write_vocab:
            f.close()
    index_to_word = dict(zip(word_to_index.values(), word_to_index.keys()))
    return count, word_to_index, index_to_word


@timeit
def get_data(words, count, word_to_index):
    """
    This function transfor the text "words" into a list
    of numbers according to the dictionary word_to_index.
    It also modifies the frequency counter 'count' to
    count the frequency of the word 'UNK'.

    :type words: list of strings
    :type count: list of tuples -> (str,int)
    :type word_to_index: dictionary
    :type index_to_word: list of dictionary
    :rtype data: list of ints
    :rtype count: list of tuples -> (str,int)
    """
    data = []
    unk_count = 0
    for word in words:
        if word in word_to_index:
            index = word_to_index[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0] = ('UNK', unk_count)
    return data, count


@timeit
def batch_generator(batch_size,
                    num_skips,
                    skip_window,
                    data_index,
                    data):

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
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    data_size = len(data)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    reading_window = deque(maxlen=span)
    for _ in range(span):
        reading_window.append(data[data_index])
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
        reading_window.append(data[data_index])
        data_index = (data_index + 1) % data_size

    return data_index, batch, labels
