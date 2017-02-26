# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import sys
import os
import os.path
import time
import math
import numpy as np
import my_txtutils as txt
tf.set_random_seed(0)

# model parameters
#
# Usage:
#   Training only:
#         Leave all the parameters as they are
#         Disable validation to run a bit faster (set validation=False below)
#         You can follow progress in Tensorboard: tensorboard --log-dir=log
#   Training and experimentation (default):
#         Keep validation enabled
#         You can now play with the parameters anf follow the effects in Tensorboard
#         A good choice of parameters ensures that the testing and validation curves stay close
#         To see the curves drift apart ("overfitting") try to use an insufficient amount of
#         training data (shakedir = "shakespeare/t*.txt" for example)
#
FLAGS = None
# SEQLEN = 30
# BATCHSIZE = 100
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
# learning_rate = 0.001  # fixed learning rate
# dropout_pkeep = 1.0    # no dropout

def main(_):

    # load data, either shakespeare, or the Python source of Tensorflow itself
    shakedir = FLAGS.text_dir
    # shakedir = "../tensorflow/**/*.py"
    codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)

    # display some stats on the data
    epoch_size = len(codetext) // (FLAGS.train_batch_size * FLAGS.seqlen)
    txt.print_data_stats(len(codetext), len(valitext), epoch_size)

    #
    # the model (see FAQ in README.md)
    #
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    # inputs
    X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, FLAGS.seqlen ]
    Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, FLAGS.seqlen, ALPHASIZE ]
    # expected outputs = same sequence shifted by 1 since we are trying to predict the next character
    Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, FLAGS.seqlen ]
    Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, FLAGS.seqlen, ALPHASIZE ]
    # input state
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    # using a NLAYERS=3 layers of GRU cells, unrolled FLAGS.seqlen=30 times
    # dynamic_rnn infers FLAGS.seqlen from the size of the inputs Xo

    onecell = rnn.GRUCell(INTERNALSIZE)
    dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
    multicell = rnn.MultiRNNCell([dropcell]*NLAYERS, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
    # Yr: [ BATCHSIZE, FLAGS.seqlen, INTERNALSIZE ]
    # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

    H = tf.identity(H, name='H')  # just to give it a name

    # Softmax layer implementation:
    # Flatten the first two dimension of the output [ BATCHSIZE, FLAGS.seqlen, ALPHASIZE ] => [ BATCHSIZE x FLAGS.seqlen, ALPHASIZE ]
    # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
    # From the readout point of view, a value coming from a cell or a minibatch is the same thing

    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x FLAGS.seqlen, INTERNALSIZE ]
    Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x FLAGS.seqlen, ALPHASIZE ]
    Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x FLAGS.seqlen, ALPHASIZE ]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x FLAGS.seqlen ]
    loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, FLAGS.seqlen ]
    Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x FLAGS.seqlen, ALPHASIZE ]
    Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x FLAGS.seqlen ]
    Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, FLAGS.seqlen ]
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
    # you can compare training and validation curves visually in Tensorboard.
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, timestamp + "-training"))
    validation_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, timestamp + "-validation"))

    # Init for saving models. They will be saved into a directory named 'checkpoints'.
    # Only the last checkpoint is kept.
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    saver = tf.train.Saver(max_to_keep=1)

    # for display: init the progress bar
    DISPLAY_FREQ = 50
    _50_BATCHES = DISPLAY_FREQ * FLAGS.train_batch_size * FLAGS.seqlen
    progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

    # init
    istate = np.zeros([FLAGS.train_batch_size, INTERNALSIZE*NLAYERS])  # initial zero input state
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0

    # training loop
    for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, FLAGS.train_batch_size, FLAGS.seqlen, nb_epochs=1000):

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, Hin: istate, lr: FLAGS.learning_rate, pkeep: FLAGS.dropout_pkeep, batchsize: FLAGS.train_batch_size}
        _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

        # save training data for Tensorboard
        summary_writer.add_summary(smm, step)

        # display a visual validation of progress (every 50 batches)
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: FLAGS.train_batch_size}  # no dropout for validation
            y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)
            txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)

        # run a validation step every 50 batches
        # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
        # so we cut it up and batch the pieces (slightly inaccurate)
        # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
        if step % _50_BATCHES == 0 and len(valitext) > 0:
            VALI_SEQLEN = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
            bsize = len(valitext) // VALI_SEQLEN
            txt.print_validation_header(len(codetext), bookranges)
            vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
            feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,  # no dropout for validation
                         batchsize: bsize}
            ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            txt.print_validation_stats(ls, acc)
            # save validation data for Tensorboard
            validation_writer.add_summary(smm, step)

        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            txt.print_text_generation_header()
            ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                print(chr(txt.convert_to_alphabet(rc)), end="")
                ry = np.array([[rc]])
            txt.print_text_generation_footer()

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            saver.save(sess, FLAGS.checkpoint_dir + '/rnn_train_' + timestamp, global_step=step)

        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)

        # loop state around
        istate = ostate
        step += FLAGS.train_batch_size * FLAGS.seqlen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text_dir',
        type=str,
        default='shakespeare/*.txt',
        help='Path to input text files.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='log',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--seqlen',
        type=int,
        default=30,
        help='How long of a sequence to consider.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--dropout_pkeep',
        type=float,
        default=1.0,
        help='What pct to keep in the dropout layers.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=50,
        help='How many text sequences to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
        How many images to test on. This test set is only used once, to evaluate
        the final accuracy of the model after training completes.
        A value of -1 causes the entire test set to be used, which leads to more
        stable results across runs.\
        """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
        How many images to use in an evaluation batch. This validation set is
        used much more often than the test set, and is an early indicator of how
        accurate the model is during training.
        A value of -1 causes the entire validation set to be used, which leads to
        more stable results across training iterations, but may be slower on large
        training sets.\
        """
    )
    parser.add_argument(
        '--print_misclassified_test_text',
        default=False,
        help="""\
        Whether to print out a list of all misclassified test text.\
        """,
        action='store_true'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help="""\
        Path to keep model checkpoints.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# all runs: FLAGS.seqlen = 30, BATCHSIZE = 100, ALPHASIZE = 98, INTERNALSIZE = 512, NLAYERS = 3
# run 1477669632 decaying learning rate 0.001-0.0001-1e7 dropout 0.5: not good
# run 1477670023 lr=0.001 no dropout: very good

# Tensorflow runs:
# 1485434262
#   trained on shakespeare/t*.txt only. Validation on 1K sequences
#   validation loss goes up from step 5M
# 1485436038
#   trained on shakespeare/t*.txt only. Validation on 5K sequences
#   On 5K sequences validation accuracy is slightly higher and loss slightly lower
#   => sequence breaks do introduce inaccuracies but the effect is small
# 1485437956
#   Trained on shakespeare/*.txt only. Validation on 1K sequences
#   On this much larger dataset, validation loss still decreasing after 6 epochs (step 35M)
# 1485440785
#   Dropout = 0.5 - Trained on shakespeare/*.txt only. Validation on 1K sequences
#   Much worse than before. Not very surprising since overfitting was not apparent
#   on the validation curves before so there is nothing for dropout to fix.

