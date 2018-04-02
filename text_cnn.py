import tensorflow as tf
from data_glove import build_dic
import numpy as np

class CNN(object):
    def __init__(self, sequence_length, num_classes, num_output, filter_size, num_filters, embedding, l2_reg_lambda=0.0):
        # placeholder for input, output and dropout
        self.input_s1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_s1')
        self.input_s2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_s2')
        self.input_tags1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_tags1')
        self.input_tags2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_tags2')
        self.input_position1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_position1')
        self.input_position2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_position2')
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.filter_sizes = filter_size
        self.num_filters = num_filters
        self.embedding = embedding
        self.l2_reg_lambda = l2_reg_lambda

        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.name_scope('embedding_layer'):
            # _, embedding = build_dic('G:\glove.6B\glove.6B.50d.txt')
            self.s_W = tf.get_variable('s_W', shape=embedding.shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(self.embedding), trainable=True)
            self.s1 = tf.nn.embedding_lookup(self.s_W, self.input_s1)
            self.s2 = tf.nn.embedding_lookup(self.s_W, self.input_s2)

            self.tags_W = tf.Variable(tf.random_uniform([45, 9], -1.0, 1.0))
            self.tags1 = tf.nn.embedding_lookup(self.tags_W, self.input_tags1)
            self.tags2 = tf.nn.embedding_lookup(self.tags_W, self.input_tags2)

            self.position_W = tf.Variable(tf.random_uniform([99, 8], -1.0, 1.0))
            self.position1 = tf.nn.embedding_lookup(self.position_W, self.input_position1)
            self.position2 = tf.nn.embedding_lookup(self.position_W, self.input_position2)

            print(self.s1.shape)
            print(self.tags1.shape)
            print(self.position1.shape)

            self.x1 = tf.concat([self.s1, self.tags1, self.position1], axis=-1)
            self.x2 = tf.concat([self.s2, self.tags2, self.position2], axis=-1)

            self.x1_expanded = tf.expand_dims(self.x1, -1)
            self.x2_expanded = tf.expand_dims(self.x2, -1)
            print(self.x1_expanded.shape)

        with tf.name_scope('conv_maxpool'):
            filter_shape = tf.convert_to_tensor(np.array([filter_size, filter_size, 1, num_filters], dtype=np.int32))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv1 = tf.nn.conv2d(
                self.x1_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv1'
            )
            conv2 = tf.nn.conv2d(
                self.x2_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv2'
            )
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu1')
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu2')
            print(h1.shape)

            pooled1 = tf.nn.max_pool(
                h1,
                ksize=[1, 1, 67 - filter_size + 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool1'
            )
            pooled2 = tf.nn.max_pool(
                h2,
                ksize=[1, 1, 67 - filter_size + 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool2'
            )

            pool_shape = pooled1.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

            self.h1_pool_flat = tf.reshape(pooled1, [-1, nodes])
            self.h2_pool_flat = tf.reshape(pooled2, [-1, nodes])

        with tf.name_scope('dropout'):
            self.h1_drop = tf.nn.dropout(self.h1_pool_flat, self.dropout_keep_prob)
            self.h2_drop = tf.nn.dropout(self.h2_pool_flat, self.dropout_keep_prob)



        with tf.name_scope('fc1'):
            self.fc1_W = tf.get_variable(
                'fc1_W',
                shape=[nodes, num_output],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.b1 = tf.Variable(tf.constant(0.1, shape=[num_output]), name='b1')
            self.output1 = tf.nn.softmax(tf.matmul(self.h1_pool_flat, self.fc1_W) + self.b1)
            self.output2 = tf.nn.softmax(tf.matmul(self.h2_pool_flat, self.fc1_W) + self.b1)
            self.fc1out = tf.concat([self.output1, self.output2], axis=-1)
            print(self.fc1out.shape)

        with tf.name_scope('fc2'):
            self.fc2_W = tf.get_variable(
                'fc2_W',
                shape=[num_output * 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.b2 = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b2'))
            self.fc2out = tf.nn.softmax(tf.matmul(self.fc1out, self.fc2_W) + self.b2)
            l2_loss += tf.nn.l2_loss(self.fc2_W)
            l2_loss += tf.nn.l2_loss(self.b2)
            self.predictions = tf.argmax(self.fc2out, 1, name='predictions')
            print(self.fc2out.shape)

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2out, labels=self.input_label)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')












