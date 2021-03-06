#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import IMAGE_SIZE, load_label_map, IMAGE_HEIGHT, IMAGE_WIDTH


def load_model_cnn(alpha=5e-5):  # `cnn` up to now

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    num_labels = len(load_label_map())
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])

    x_image = tf.reshape(x, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])  # input layer

    # First Convolutional Layer
    conv_layer1_weight = weight_variable([5, 5, 1, 32])  # kernel_size 5x5
    conv_layer1_bias = bias_variable([32])
    pool_layer1 = max_pool_2x2(
        tf.nn.relu(
            conv2d(x_image, conv_layer1_weight) + conv_layer1_bias
        )
    )  # poll_layer1 50% reduction of x_image in_width,in_height

    # Second Convolutional Layer
    conv_layer2_weight = weight_variable([5, 5, 32, 64])
    conv_layer2_bias = bias_variable([64])
    pool_layer2 = max_pool_2x2(
        tf.nn.relu(
            conv2d(pool_layer1, conv_layer2_weight) + conv_layer2_bias
        )
    )  # poll_layer2 50% reduction of pool_layer1 in_width,in_height

    # Fully Connected Layer
    fc_layer_weight = weight_variable([IMAGE_HEIGHT // 4 * IMAGE_WIDTH // 4 * 64, 1024])
    fc_layer_bias = bias_variable([1024])

    pool_layer2_flat = tf.reshape(pool_layer2, [-1, IMAGE_HEIGHT // 4 * IMAGE_WIDTH // 4 * 64])
    fc_layer = tf.nn.relu(tf.matmul(pool_layer2_flat, fc_layer_weight) + fc_layer_bias)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    fc_layer_drop = tf.nn.dropout(fc_layer, keep_prob)

    # Readout Layer
    output_layer_weight = weight_variable([1024, num_labels])
    output_layer_bias = bias_variable([num_labels])

    y_conv = tf.add(tf.matmul(fc_layer_drop, output_layer_weight),
                    output_layer_bias)

    y = tf.placeholder(tf.float32, shape=[None, num_labels])
    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
        )
        tf.summary.scalar('loss_function', loss)
    with tf.name_scope('learing_rate'):
        optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    prediction = tf.argmax(y_conv, 1)  # for recognise
    saver = tf.train.Saver(max_to_keep=1)
    model = {'x': x,
             'y': y,
             'optimizer': optimizer,
             'loss': loss,
             'keep_prob': keep_prob,
             'accuracy': accuracy,
             'prediction': prediction,
             'merged': merged,
             'saver': saver,
             }

    return model


def load_model_rnn(alpha=5e-5):  # RNN with LSTM support
    from tensorflow.contrib import rnn
    # Parameters
    learning_rate = alpha  # alpha value

    # Network Parameters
    n_input = IMAGE_WIDTH  # MNIST data input
    n_steps_each_sample = IMAGE_HEIGHT  # timesteps
    n_hidden = 128  # hidden layer num of features
    n_classes = len(load_label_map())  # num of classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_steps_each_sample, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def rnn_func(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # (batch_size, n_steps_each_sample, n_input) -> (batch_size, n_input)
        # Unstack to get a list of 'n_shape' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps_each_sample, 1)
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop
        return outputs[-1] @ weights['out'] + biases['out']  # reload `+`

    pred = rnn_func(x, weights, biases)

    # Define loss and optimizer
    with tf.name_scope('loss_function'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar('loss_function', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    prediction = tf.argmax(pred, 1)  # for recognise
    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=1)
    model = {'x': x,
             'y': y,
             'optimizer': optimizer,
             'loss': cost,
             'keep_prob': keep_prob,
             'accuracy': accuracy,
             'prediction': prediction,
             'merged': merged,
             'saver': saver,
             }
    return model
