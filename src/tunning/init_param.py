import os
import sys
from random import randint
import numpy as np
import inspect
from mpl_toolkits.mplot3d import Axes3D
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

car1 = np.random.random_sample([10]) + 1
car2 = np.random.random_sample([10]) + 1
car3 = np.random.random_sample([10])
cdr1 = np.zeros([10])
cdr2 = np.zeros([10])
INIT_PARAM = list(zip(car1, cdr1))
par2 = list(zip(cdr2, car2))
par3 = list(zip(car3, car3))
INIT_PARAM.extend(par2)
INIT_PARAM.extend(par3)
INIT_PARAM.append((1.0, 1.0))

number_of_exp = len(INIT_PARAM)
my_xs = np.array([x for (x, y) in INIT_PARAM])
my_ys = np.array([y for (x, y) in INIT_PARAM])

results = []
info = []
for i, pa in enumerate(INIT_PARAM):
    print("\n ({0} of {1})".format(i + 1, number_of_exp))
    config = wv.Config(init_param=pa)
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

my_zs = np.array(results)
best_result = max(list(zip(results, INIT_PARAM, info)))
result_string = """In an experiment with {0} init params
the best one is {1} with score = {2}.
\n INFO = {3}""".format(number_of_exp,
                        best_result[1],
                        best_result[0],
                        best_result[2])

file = open("init_param.txt", "w")
file.write(result_string)
file.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(my_xs, my_ys, my_zs, c="r", marker='^')

ax.set_xlabel('init param 1')
ax.set_ylabel('init param 2')
ax.set_zlabel('score')

plt.savefig("init_param.png")
