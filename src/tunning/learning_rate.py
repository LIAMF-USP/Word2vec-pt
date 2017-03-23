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
file_path = os.path.join(file_path, "pt96.txt")
eval_path = os.path.join(parentdir, "evaluation")
eval_path = os.path.join(eval_path, "AnalogiesBr.txt")

my_data = DataReader(file_path)
my_data.get_data()
word2index = my_data.word2index
index2word = my_data.index2word

number_of_exp = 20
part1 = int(number_of_exp/2)
part2 = number_of_exp - part1
lr1 = np.random.random_sample([part1]) / 10
lr2 = np.random.random_sample([part2])
LEARNING_RATE = np.concatenate((lr1, lr2))
LEARNING_RATE.sort()
results = []

for i, learning_rate in enumerate(LEARNING_RATE):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    config = wv.Config(lr=learning_rate)
    my_model = wv.SkipGramModel(config)
    embeddings = wv.run_training(my_model,
                                 my_data,
                                 verbose=False,
                                 visualization=False,
                                 debug=False)
    score, _ = util.score(index2word,
                          word2index,
                          embeddings,
                          eval_path,
                          verbose=False)
    results.append(score)

LEARNING_RATE = list(LEARNING_RATE)
best_result = max(list(zip(results, LEARNING_RATE)))
result_string = """In an experiment with {0} learning rate values
the best one is {1} with score = {2}.""".format(number_of_exp,
                                                best_result[1],
                                                best_result[0])

file = open("learning_rate.txt", "w")
file.write(result_string)
file.close()


plt.plot(LEARNING_RATE, results)
plt.xscale('log')
plt.xlabel("learning_rate")
plt.ylabel("score")
plt.savefig("learning_rate.png")
