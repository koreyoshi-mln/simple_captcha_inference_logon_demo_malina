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
from dataset import DataSet

try:
    FileNotFoundError
except NameError:
    # py2
    FileNotFoundError = IOError

formatted_dataset_dir = trainer_dir


def train(alpha=5e-5, nn_type=NNType.cnn, target_accuracy=0.9955):

    label_map_path = os.path.join(formatted_dataset_dir, 'label_map.pickle')
    formatted_train_dataset_path = os.path.join(formatted_dataset_dir, 'train_dataset.pickle')
    formatted_test_dataset_path = os.path.join(formatted_dataset_dir, 'test_dataset.pickle')

    def _compat_pickle_load(path):
        with open(path, 'rb') as f:
            import sys
            if sys.version_info.major == 3:
                obj = pickle.load(f, encoding='latin1')
            else:
                obj = pickle.load(f)

        return obj

    print("loading %s" % label_map_path)
    label_map = _compat_pickle_load(label_map_path)

    print("load %s" % formatted_train_dataset_path)
    train_dataset = _compat_pickle_load(formatted_train_dataset_path)
    train_labels = train_dataset.labels

    print("load %s" % formatted_test_dataset_path)
    test_dataset = _compat_pickle_load(formatted_test_dataset_path)
    test_data, test_labels = test_dataset.images, test_dataset.labels

    if nn_type == NNType.rnn:
        test_data = test_data.reshape((len(test_data), IMAGE_HEIGHT, IMAGE_WIDTH))

    num_labels = len(label_map)

    print("train_data:", train_dataset.images.shape)
    print("train_labels:", train_labels.shape)
    print("test_data:", test_data.shape)
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

    batch_size = 128

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
            with open(formatted_train_dataset_path, 'wb') as f:
                pickle.dump(train_dataset, f, protocol=2)

        graph_log_dir = model_ckpt_dir
        writer = tf.summary.FileWriter(graph_log_dir)

        while True:
            batch_data, batch_labels = train_dataset.next_batch(batch_size)
            if nn_type == NNType.rnn:
                batch_data = batch_data.reshape((len(batch_data), IMAGE_HEIGHT, IMAGE_WIDTH))

            if step % 10 == 0:  # Display, Test and Save

                summary, acc_train, loss, _ = sess.run(
                    [merged, accuracy, cost, optimizer],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels,
                        keep_prob: 0.5
                    }
                )
                acc_test = sess.run(
                      accuracy,
                      feed_dict={
                          x: test_data,
                          y: test_labels,
                          keep_prob: 1.0
                      })

                writer.add_summary(summary, step)
                print("step %4d, train_accuracy: %.4f, loss: %.4f test_accuracy: %.4f" %
                      (step, acc_train, loss, acc_test))

                # Test Whether you can exit
                if acc_test > target_accuracy or loss < 0.002:
                    print('training done.')
                    save_model(step)
                    break

                if step % 100 == 0:  # save the model every 100 step
                    save_model(step)

            else:
                loss, _ = sess.run(
                    [cost, optimizer],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels,
                        keep_prob: 0.5
                    }
                )

            step += 1




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
