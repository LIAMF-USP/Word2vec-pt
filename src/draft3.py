# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import inspect
import string
import time
import util
from datareader import DataReader


class Config(object):
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self,
                 vocab_size=50000,
                 batch_size=128,
                 embed_size=128,
                 skip_window=1,
                 num_skips=2,
                 num_sampled=64,
                 lr=1.0,
                 num_steps=100001,
                 skip_step=2000,
                 valid_size=16,
                 valid_window=100):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.lr = lr
        self.num_steps = num_steps
        self.skip_step = skip_step
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.array(random.sample(range(self.valid_window),
                                                     self.valid_size))


class SkipGramModel:
    """
    Build the graph for word2vec model
    """
    def __init__(self, config):
        self.logdir = util.newlogname()
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.embed_size = self.config.embed_size
        self.batch_size = self.config.batch_size
        self.num_sampled = self.config.num_sampled
        self.lr = self.config.lr
        self.valid_examples = self.config.valid_examples
        self.build_graph()

    def create_placeholders(self):
            """
            Creat placeholder for the models graph
            """
            with tf.name_scope("words"):
                self.center_words = tf.placeholder(tf.int32,
                                                   shape=[self.batch_size],
                                                   name='center_words')
                self.targets = tf.placeholder(tf.int32,
                                              shape=[self.batch_size, 1],
                                              name='target_words')
                self.valid_dataset = tf.constant(self.valid_examples,
                                                 dtype=tf.int32)

    def create_weights(self):
        """
        Creat all the weigs and bias for the models graph
        """
        emshape = (self.vocab_size, self.embed_size)
        eminit = tf.random_uniform(emshape, -1.0, 1.0)
        self.embeddings = tf.Variable(eminit, name="embeddings")

        with tf.name_scope("softmax"):
                    Wshape = (self.vocab_size, self.embed_size)
                    bshape = (self.vocab_size)
                    std = 1.0/(self.config.embed_size ** 0.5)
                    Winit = tf.truncated_normal(Wshape, stddev=std)
                    binit = tf.zeros(bshape)
                    self.weights = tf.get_variable("weights",
                                                   dtype=tf.float32,
                                                   initializer=Winit)
                    self.biases = tf.get_variable("biases",
                                                  dtype=tf.float32,
                                                  initializer=binit)

    def create_loss(self):
        with tf.name_scope("loss"):
            self.embed = tf.nn.embedding_lookup(self.embeddings,
                                                self.center_words,
                                                name='embed')
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights,
                                                                  self.biases,
                                                                  self.targets,
                                                                  self.embed,
                                                                  self.num_sampled,
                                                                  self.vocab_size))

    def create_optimizer(self):
        with tf.name_scope("train"):
            opt = tf.train.AdagradOptimizer(self.lr)
            self.optimizer = opt.minimize(self.loss)

    def create_valid(self):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),
                                     1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                  self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings,
                                    tf.transpose(self.normalized_embeddings))

    def create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
            """
            Build the graph for our model
            """
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.create_placeholders()
                self.create_weights()
                self.create_loss()
                self.create_optimizer()
                self.create_valid()
                self.create_summaries()

filename = 'pt96.txt'
file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
file_path = os.path.join(file_path, 'data')
file_path = os.path.join(file_path, filename)

my_data = DataReader(file_path)
vocab_size = 50000
my_data.get_data(vocab_size)
my_model = SkipGramModel(Config())


def run_training(model, data, verbose=True):
    logdir = model.logdir
    batch_size = model.config.batch_size
    num_skips = model.config.num_skips
    skip_window = model.config.skip_window
    valid_examples = model.config.valid_examples
    num_steps = model.config.num_steps
    data_index = 0
    with tf.Session(graph=model.graph) as session:
        tf.global_variables_initializer().run()
        ts = time.time()
        print("Initialized")
        print("\n&&&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&&&")
        print("\ntensorboard  --logdir={}\n".format(logdir))
        average_loss = 0
        writer = tf.summary.FileWriter(logdir, session.graph)
        for step in range(num_steps):
            data_index, batch_data, batch_labels = data.batch_generator(batch_size,
                                                                       num_skips,
                                                                       skip_window,
                                                                       data_index)
            feed_dict = {model.center_words: batch_data,
                         model.targets: batch_labels}
            _, l, summary = session.run([model.optimizer,
                                         model.loss,
                                         model.summary_op], feed_dict=feed_dict)
            average_loss += l
            writer.add_summary(summary, global_step=step)
            writer.flush()
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    print("Average loss at step", step, ":", average_loss)
                    average_loss = 0

            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0 and verbose:
                sim = model.similarity.eval()
                for i in range(model.config.valid_size):
                    valid_word = data.index_to_word[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = data.index_to_word[nearest[k]]
                        log = "%s %s," % (log, close_word)
                    print(log)

        final_embeddings = model.normalized_embeddings.eval()
    te = time.time()
    return final_embeddings, te-ts

_, duration = run_training(my_model, my_data)
print("duration= ", duration)

# duration=  262.3832314014435
# [Finished in 284.3s]
