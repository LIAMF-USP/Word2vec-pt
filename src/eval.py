import pickle
import util
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-f",
                    "--file",
                    type=str,
                    default='None',
                    help="""pickle file to apply
                    the evaluation function (default=None)""")

parser.add_argument("-l",
                    "--language",
                    type=str,
                    default='pt',
                    help="""language selection: pt, en (default=pt)""")


args = parser.parse_args()
file_path = args.file
with open(file_path, "rb") as s:
    d = pickle.load(s)
    pass


embeddings = d['embeddings']
word2index = d['word2index']
index2word = d['index2word']

if args.language == 'pt':
  eval_path = "./evaluation/questions-words-ptbr.txt"
else:
  eval_path = "./evaluation/questions-words.txt"
score, report = util.score(index2word,
                           word2index,
                           embeddings,
                           eval_path,
                           verbose=True,
                           raw=True)
print()
print("Score = {}".format(score))
for result in report:
    print(result)
