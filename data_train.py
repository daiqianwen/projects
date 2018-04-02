import tensorflow as tf
import time
import datetime
import os
import numpy as np
import data_glove
from text_cnn import CNN
from tensorflow.contrib import learn
from data_glove import build_dic


# Parameters
# ============================================================================

# Data loading parameters
tf.flags.DEFINE_float("train_sample_percentage", .9, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string('data_file', '.\\file\\tt.txt', 'Data source')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 67, 'dimension of character embedding (default: 50)')
tf.flags.DEFINE_integer("filter_size", 3, "Comma-separated filter sizes (default: '3)")
tf.flags.DEFINE_integer('sent_length', 102, 'sentence length (default: 50)')
tf.flags.DEFINE_integer('num_output', 128, 'num outout')
tf.flags.DEFINE_integer('num_classes', 2, 'num classes (default: 2)')
tf.flags.DEFINE_integer('num_filters', 512, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 1.0, 'L2 regularization lambda (default: 0.0)')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size (default: 64)')
tf.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on dev sets after many steps (default: 100)')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'Save model after this many steps (default: 100)')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default: 5)')

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print("")

# Data preparation
# ==================================================================================

# Load data
print('Loading data...')
s1_pad, s2_pad, tags1, tags2, relative_position1, relative_position2, label = data_glove.read_data_sets1(FLAGS.data_file)

# Build vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.sent_length)
tags1_pad = np.array(list(vocab_processor.fit_transform(tags1)))
tags2_pad = np.array(list(vocab_processor.fit_transform(tags2)))
relative_position1_pad = np.array(list(vocab_processor.fit_transform(relative_position1)))
relative_position2_pad = np.array(list(vocab_processor.fit_transform(relative_position2)))

# s1, s2, label = data_glove.read_data_sets(FLAGS.data_file)
sample_num = len(label)
train_end = int(sample_num * FLAGS.train_sample_percentage)

# Split train/test data
s1_train, s1_test = s1_pad[:train_end], s1_pad[train_end:]
s2_train, s2_test = s2_pad[:train_end], s2_pad[train_end:]
tags1_train, tags1_test = tags1_pad[:train_end], tags1_pad[train_end:]
tags2_train, tags2_test = tags2_pad[:train_end], tags2_pad[train_end:]
relative_position1_train, relative_position1_test = relative_position1_pad[:train_end], relative_position1_pad[train_end:]
relative_position2_train, relative_position2_test = relative_position2_pad[:train_end], relative_position2_pad[train_end:]
label_train, label_test = label[:train_end], label[train_end:]
print('train/test split: {:d}/{:d}'.format(len(label_train), len(label_test)))
# print(label_train[0])

_, embedding = build_dic('.\\file\\glove.6B.50d.txt')

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            sequence_length=FLAGS.sent_length,
            num_classes=FLAGS.num_classes,
            num_output=FLAGS.num_output,
            filter_size=FLAGS.filter_size,
            num_filters=FLAGS.num_filters,
            embedding=embedding,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # train_step = tf.train.AdamOptimizer(0.001).minimize(cnn.loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        # Initial all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        def train_step(s1_batch, s2_batch, tags1_batch, tags2_batch, position1_batch, position2_batch, label_batch):
            # a single training step
            feed_dict = {
                cnn.input_s1: s1_batch,
                cnn.input_s2: s2_batch,
                cnn.input_tags1: tags1_batch,
                cnn.input_tags2: tags2_batch,
                cnn.input_position1: position1_batch,
                cnn.input_position2: position2_batch,
                cnn.input_label: label_batch
            }
            _, step, predictions, loss, accuracy = sess.run(
                [train_op, global_step, cnn.predictions, cnn.loss, cnn.accuracy],
                feed_dict
            )
            print('step{}, loss{:g}, acc{:g}'.format(step, loss, accuracy))
            # print(predictions)

        STS_train = data_glove.dataset(s1_train, s2_train, tags1_train, tags2_train,
                                       relative_position1_train, relative_position2_train, label_train)

        for i in range(1000):
            batch_train = STS_train.next_batch(FLAGS.batch_size)
            # print(batch_train.shape)
            train_step(batch_train[0], batch_train[1], batch_train[2], batch_train[3],
                       batch_train[4], batch_train[5], batch_train[6])
            if i % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                train_step(s1_test, s2_test, tags1_test, tags2_test, relative_position1_test, relative_position2_test,
                           label_test)
                print("")

















