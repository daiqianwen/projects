import tensorflow as tf
import numpy as np
from data_glove import build_dic

class TextCNN(object):
    def __init__(self, sentence_length, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # placeholder for input, output and dropout
        self.input_s1 = tf.placeholder(dtype=tf.int32, shape=[None, sentence_length], name='input_s1')
        self.input_s2 = tf.placeholder(dtype=tf.int32, shape=[None, sentence_length], name='input_s2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

        l2_loss = tf.constant(0.0)

        # self.init_weight()

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            _, word_embedding = build_dic('G:\glove.6B\glove.6B.50d.txt')
            embedding_size = word_embedding.shape[1]
            self.W = tf.get_variable('W', shape=word_embedding.shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(word_embedding), trainable=True)
            self.s1 = tf.nn.embedding_lookup(self.W, self.input_s1)
            self.s2 = tf.nn.embedding_lookup(self.W, self.input_s2)
            self.x = tf.concat([self.s1, self.s2], axis=1)
            self.x = tf.expand_dims(self.x, -1)

        # create a convolution layer and maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s'%filter_size):
                # convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name='b')
                conv = tf.nn.conv2d(
                    self.x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='Valid',
                    name='conv'
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Max pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sentence_length*2-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='Valid',
                    name='pooled'
                )

                pooled_outputs.append(pooled)

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, float), name='accuracy')









