from util import timeit
import string
from collections import Counter
import inspect
import os


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
