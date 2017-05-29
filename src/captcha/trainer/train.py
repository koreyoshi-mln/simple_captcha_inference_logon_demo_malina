#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from argparse import ArgumentParser

try:
    import cPickle as pickle
except ImportError:
    import pickle

import tensorflow as tf

trainer_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(trainer_dir)
sys.path.append(home_dir)

import common.load_model_nn as load_nn
from common.common import find_model_ckpt, IMAGE_WIDTH, IMAGE_HEIGHT, NNType

try:
    FileNotFoundError
except NameError:
    # py2
    FileNotFoundError = IOError

formatted_dataset_path = os.path.join(trainer_dir, 'formatted_dataset.pickle')


def train(alpha=5e-5, nn_type=NNType.cnn, target_accuracy=0.9955):
    print("loading %s..." % formatted_dataset_path)
    with open(formatted_dataset_path, 'rb') as f:
        import sys
        if sys.version_info.major == 3:
            save = pickle.load(f, encoding='latin1')
        else:
            save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        label_map = save['label_map']

        if nn_type == NNType.rnn:
            train_dataset = train_dataset.reshape((len(train_dataset), IMAGE_HEIGHT, IMAGE_WIDTH))
            test_dataset = test_dataset.reshape((len(test_dataset), IMAGE_HEIGHT, IMAGE_WIDTH))

    num_labels = len(label_map)

    print("train_dataset:", train_dataset.shape)
    print("train_labels:", train_labels.shape)
    print("test_dataset:", test_dataset.shape)
    print("test_labels:", test_labels.shape)
    print("num_labels:", num_labels)

    if nn_type == NNType.cnn:
        model = load_nn.load_model_cnn(alpha=alpha)
    else:
        model = load_nn.load_model_rnn(alpha=alpha)

    x = model['x']
    y = model['y']
    cost = model['loss']
    optimizer = model['optimizer']
    accuracy = model['accuracy']
    keep_prob = model['keep_prob']
    merged = model['merged']
    saver = model['saver']
    #graph = model['graph']

    batch_size = 64

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        model_ckpt_path, origin_step, ok = find_model_ckpt(nn_type=nn_type)  # try to continue ....
        model_ckpt_dir = os.path.dirname(model_ckpt_path)
        step = origin_step
        if not ok:
            print("Initialized")
        else:  # try continue to train
            saver.restore(sess, model_ckpt_path)
            print('found %s, step from %d' % (model_ckpt_path, step))

        def save_model(_step):
            saver.save(
                sess,
                os.path.join(model_ckpt_dir, 'model'),
                global_step=_step
            )

        graph_log_dir = model_ckpt_dir
        writer = tf.summary.FileWriter(graph_log_dir)

        while True:
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            sess.run(
                [optimizer, cost],
                feed_dict={
                    x: batch_data,
                    y: batch_labels,
                    keep_prob: 0.5
                }
            )
            step += 1
            if step % 10 == 0:  # Display, Test and Save
                summary, acc, loss = sess.run(
                    [merged, accuracy, cost],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels,
                        keep_prob: 1.0
                    }
                )
                writer.add_summary(summary, step)

                print("step %4d, accuracy: %.4f, loss: %.2f" % (step, acc, loss))

                if acc > target_accuracy:  # Test Whether you can exit
                    print('you can re-format dataset '
                          'to continue training')
                    print('training done.')
                    save_model(step)
                    break

                if step % 100 == 0:  # save the model every 100 step
                    save_model(step)

            else:
                summary, train_accuracy = sess.run(
                    [merged, accuracy],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels,
                        keep_prob: 1.0
                    }
                )
                writer.add_summary(summary, step)

        print("Test accuracy: %g" %
              sess.run(
                  accuracy,
                  feed_dict={
                      x: test_dataset,
                      y: test_labels,
                      keep_prob: 1.0
                  })
              )


def cli():
    parser = ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default='5e-5',
                        help='convergence rate for train default 5e-5')

    parser.add_argument('-nn', '--nn-type',  default='cnn', choices=['cnn', 'rnn'],
                        help='select neural network model type')

    parser.add_argument('-acc', '--accuracy', dest='target_accuracy', type=float, default=0.9955,
                        help='target accuracy, default 0.9955')

    kwargs = parser.parse_args().__dict__
    if kwargs['nn_type'] == 'cnn':
        kwargs['nn_type'] = NNType.cnn
    elif kwargs['nn_type'] == 'rnn':
        kwargs['nn_type'] = NNType.rnn

    # print(kwargs)
    train(**kwargs)


if __name__ == '__main__':
    cli()
