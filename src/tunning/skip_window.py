import os
import sys
from random import randint
import numpy as np
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datareader import DataReader
import word2vec as wv
import util

file_path = os.path.join(parentdir, "data")
file_path = os.path.join(file_path, "Wiki.txt")
eval_path = os.path.join(parentdir, "evaluation")
eval_path = os.path.join(eval_path, "questions-words-ptbr.txt")

my_data = DataReader(file_path)
my_data.get_data()
word2index = my_data.word2index
index2word = my_data.index2word

SKIP_WINDOW = [1,
               2,
               3,
               4,
               5,
               6]
number_of_exp = len(SKIP_WINDOW)
results = []
info = []
batch = 60

for i, sk in enumerate(SKIP_WINDOW):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    nk = 2 * sk
    batch = batch - (batch % nk)
    config = wv.Config(skip_window=sk, batch_size=batch, num_skips=nk)
    attrs = vars(config)
    config_info = ["%s: %s" % item for item in attrs.items()]
    info.append(config_info)
    my_model = wv.SkipGramModel(config)
    embeddings = wv.run_training(my_model,
                                 my_data,
                                 verbose=False,
                                 visualization=False,
                                 debug=False)
    score, report = util.score(index2word,
                               word2index,
                               embeddings,
                               eval_path,
                               verbose=False,
                               raw=True)
    results.append(score)
    print("Score = {}".format(score))
    for result in report:
        print(result)

best_result = max(list(zip(results, SKIP_WINDOW, info)))
result_string = """In an experiment with {0} skip windows
the best window size is {1} with score = {2}
\n INFO = {3}""".format(number_of_exp,
                        best_result[1],
                        best_result[0],
                        best_result[2])

file = open("skip_window.txt", "w")
file.write(result_string)
file.close()

plt.plot(SKIP_WINDOW, results)
plt.xlabel("skip window")
plt.ylabel("score")
plt.savefig("skip_window.png")
