'''
Deep Learning Programming Assignment 4
--------------------------------------
Name: Surjodoy Ghosh Dastider
Roll No.: 16CS60R75
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq

# parses the dataset
import ptb_reader
import argparse

parser = argparse.ArgumentParser(description='Program Description')
parser.add_argument('-t', '--test', help='Test Path Argument', required=False)
args = vars(parser.parse_args())


# define artifact directories where results from the session can be saved
model_path = "./model.ckpt"

# load dataset
train_data, valid_data, test_data, _ = ptb_reader.ptb_raw_data("../data")
if args['test']:
	train_path = "../data/ptb.train.txt"
	test_path = args['test']
	word_to_id = ptb_reader._build_vocab(train_path)
	test_data = ptb_reader._file_to_word_ids(test_path, word_to_id)

class PTBModel(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_data")
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps], name="targets")

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=False)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # initializer used for reusable variable initializer (see `get_variable`)
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], initializer=initializer)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        states = []
        state = self.initial_state

        with tf.variable_scope("RNN", initializer=initializer):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                inputs_slice = inputs[:,time_step,:]
                (cell_output, state) = cell(inputs_slice, state)

                outputs.append(cell_output)
                states.append(state)

        self.final_state = states[-1]

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        w = tf.get_variable("softmax_w", [size, vocab_size], initializer=initializer)
        b = tf.get_variable("softmax_b", [vocab_size], initializer=initializer)

        logits = tf.nn.xw_plus_b(output, w, b) # compute logits for loss
        targets = tf.reshape(self.targets, [-1]) # reshape our target outputs
        weights = tf.ones([batch_size * num_steps]) # used to scale the loss average

        # computes loss and performs softmax on our fully-connected output layer
        loss = legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights], vocab_size)
        self.cost = cost = tf.div(tf.reduce_sum(loss), batch_size, name="cost")

        if is_training:
            # setup learning rate variable to decay
            self.lr = tf.Variable(1.0, trainable=False)

            # define training operation and clip the gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), name="train")
        else:
            # if this model isn't for training (i.e. testing/validation) then we don't do anything here
            self.train_op = tf.no_op()

def run_epoch(sess, model, data):
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()

    # accumulated counts
    costs = 0.0
    iters = 0

    # initial RNN state
    state = model.initial_state.eval()

    for step, (x, y) in enumerate(ptb_reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        cost, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict={
            model.input_data: x,
            model.targets: y,
            model.initial_state: state
        })
        costs += cost
        iters += model.num_steps

        perplexity = np.exp(costs / iters)

        if step % 100 == 0:
            break

    return (costs / iters), perplexity

class Config(object):
    batch_size = 50
    num_steps = 30 # number of unrolled time steps
    hidden_size = 450 # number of blocks in an LSTM cell
    vocab_size = 10000
    max_grad_norm = 5 # maximum gradient for clipping
    init_scale = 0.05 # scale between -0.1 and 0.1 for all random initialization
    keep_prob = 0.5 # dropout probability
    num_layers = 3 # number of LSTM layers
    learning_rate = 1.0
    lr_decay = 0.8
    lr_decay_epoch_offset = 6 # don't decay until after the Nth epoch

# default settings for training
train_config = Config()

# our test run uses a batch size and time step of one
eval_config = Config()
eval_config.batch_size = 1
eval_config.num_steps = 1

# number of epochs to perform over the training data
num_epochs = 5

with tf.Session() as sess:
    # define our training model
    with tf.variable_scope("model", reuse=None):
        train_model = PTBModel(is_training=True, config=train_config)

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    # initialize our variables
    sess.run(tf.global_variables_initializer())

    train_costs = []
    train_perps = []
	
#    for i in range(num_epochs):
#        print("Epoch: %d Learning Rate: %.3f" % (i + 1, sess.run(train_model.lr)))

#        # run training pass
#        train_cost, train_perp = run_epoch(sess, train_model, train_data)
#        print("Epoch: %i Training Perplexity: %.3f (Cost: %.3f)" % (i + 1, train_perp, train_cost))
#        train_costs.append(train_cost)
#        train_perps.append(train_perp)

#        saver.save(sess, model_path)
#	saver.save(sess, model_path)
	
	
    # run test pass
    os.system('wget https://github.com/ZTK13/LSTM-PennTreebank/blob/master/model.ckpt.data-00000-of-00001?raw=true -O model.ckpt.data-00000-of-00001')
    os.system('wget https://github.com/ZTK13/LSTM-PennTreebank/blob/master/model.ckpt.index?raw=true -O model.ckpt.index')
    os.system('wget https://github.com/ZTK13/LSTM-PennTreebank/blob/master/model.ckpt.meta?raw=true -O model.ckpt.meta')
    os.system('wget https://github.com/ZTK13/LSTM-PennTreebank/raw/master/checkpoint -O checkpoint')
    saver.restore(sess, model_path)
    
    # we create a separate model for validation and testing to alter the batch size and time steps
    # reuse=True reuses variables from the previously defined `train_model`
    with tf.variable_scope("model", reuse=True):
        test_model = PTBModel(is_training=False, config=eval_config)
    
    test_cost, test_perp = run_epoch(sess, test_model, test_data)
    print("Test Perplexity: %.3f (Cost: %.3f)" % (test_perp, test_cost))


