import util
from data_process import read_text, build_vocab, get_data, batch_generator
import tensorflow as tf
import numpy as np
from random import sample


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
                 num_steps=100000,
                 skip_step=2000):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.lr = lr
        self.num_steps = num_steps
        self.skip_step = skip_step


class SkipGramModel:
    """
    Build the graph for word2vec model
    """
    def __init__(self, config):
        self.name = newlogname()
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.embed_size = self.config.embed_size
        self.batch_size = self.config.batch_size
        self.num_sampled = self.config.num_sampled
        self.lr = self.config.lr
        self.data = self.config.data
        self.word2index = self.config.word2index
        self.index_to_word = self.config.index_to_word
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.array(sample(range(self.valid_window), self.valid_size))
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
