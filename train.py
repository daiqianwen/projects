import tensorflow as tf
import time
import datetime
import os
import numpy as np
import data_glove
from CNN_model import TextCNN
from tensorflow.contrib import learn

# Parameters
# ============================================================================

# Data loading parameters
tf.flags.DEFINE_float("train_sample_percentage", .9, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string('data_file', 'G:\\ldc17\\tt.txt', 'Data source')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 50, 'dimension of character embedding (default: 50)')
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer('sent_length', 50, 'sentence length (default: 50)')
tf.flags.DEFINE_integer('num_classes', 2, 'num classes (default: 2)')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 1.0, 'L2 regularization lambda (default: 0.0)')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size (default: 64)')
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

s1, s2, label = data_glove.read_data_sets(FLAGS.data_file)
sample_num = len(label)
train_end = int(sample_num * FLAGS.train_sample_percentage)

# Split train/test data
s1_train, s1_test = s1[:train_end], s1[train_end:]
s2_train, s2_test = s2[:train_end], s2[train_end:]
label_train, label_test = s1[:train_end], s2[train_end:]
print('train/test split: {:d}/{:d}'.format(len(label_train), len(label_test)))

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sentence_length=FLAGS.sent_length,
            num_classes=FLAGS.num_classes,
            filter_sizes=FLAGS.filter_sizes,
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # Define training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity(optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print('Writing to {}\n'.format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initial all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        def train_step(s1, s2, label):
            # a single training step
            feed_dict = {
                cnn.input_s1: s1,
                cnn.input_s2: s2,
                cnn.input_y: label,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print('{}: step{}, loss{:g}, acc{:g}'.format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def test_step(s1, s2, label, writer=None):
            feed_dict = {
                cnn.input_s1: s1,
                cnn.input_s2: s2,
                cnn.input_y: label,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, test_summary_op, cnn.loss, cnn.accuracy],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        STS_train = data_glove.dataset(s1=s1_train, s2=s2_train, label=label_train)
        # Training lop, for each batch

        for i in range(40000):
            batch_train = STS_train.next_batch(FLAGS.batch_size)

            train_step(batch_train[0], batch_train[1], batch_train[2])
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print('\nEvaluation:')
                test_step(s1_test, s2_test, label_test, writer=test_summary_writer)
                print('')
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))






















