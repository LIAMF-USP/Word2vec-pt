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
eval_path = os.path.join(eval_path, "questions-words-ptbr.txt")

my_data = DataReader(file_path)
my_data.get_data()
word2index = my_data.word2index
index2word = my_data.index2word

EMB_SIZE = np.array(range(1, 51)) * 10
number_of_exp = len(EMB_SIZE)
results = []
reports = []
for i, em in enumerate(EMB_SIZE):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    config = wv.Config(embed_size=em)
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
    reports.append(report)
    results.append(score)

EMB_SIZE = list(EMB_SIZE)
best_result = max(list(zip(results, EMB_SIZE, reports)))
result_string = """In an experiment with {0} embeddings sizes
the best size is {1} with score = {2}.""".format(number_of_exp,
                                                 best_result[1],
                                                 best_result[0])

file = open("emb_size.txt", "w")
file.write(result_string)
file.write(" ")
for sta in best_result[2]:
    print(sta)
file.close()


plt.plot(EMB_SIZE, results)
plt.xlabel("embedding size")
plt.ylabel("score")
plt.savefig("emb_size.png")
