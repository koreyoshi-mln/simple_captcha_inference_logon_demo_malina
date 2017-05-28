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

from common.load_model_nn import load_model_nn
from common.common import find_model_ckpt

try:
    FileNotFoundError
except NameError:
    # py2
    FileNotFoundError = IOError

formatted_dataset_path = os.path.join(trainer_dir, 'formatted_dataset.pickle')
ckpt_dir = os.path.join(trainer_dir, '.checkpoint')
# graph_log_dir = os.path.join(trainer_dir, 'logs')
graph_log_dir = ckpt_dir


def train(alpha=5e-5):
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

    num_labels = len(label_map)

    print("train_dataset:", train_dataset.shape)
    print("train_labels:", train_labels.shape)
    print("test_dataset:", test_dataset.shape)
    print("test_labels:", test_labels.shape)
    print("num_labels:", num_labels)

    model = load_model_nn(alpha)
    x = model['x']
    y = model['y']
    loss = model['loss']
    optimizer = model['optimizer']
    accuracy = model['accuracy']
    keep_prob = model['keep_prob']
    merged = model['merged']
    saver = model['saver']
    graph = model['graph']

    print("checkpoint saved dir: ", ckpt_dir)

    batch_size = 64

    def save_model(_step):
        saver.save(
            session,
            os.path.join(ckpt_dir, 'weibo.cn-model.ckpt'),
            global_step=_step
        )

    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(graph_log_dir, session.graph)
        tf.global_variables_initializer().run()

        step = 0
        try:
            model_ckpt_path, global_step = find_model_ckpt()  # try to continue ....
        except FileNotFoundError:
            print("Initialized")
        else:  # try continue to train
            saver.restore(session, model_ckpt_path)
            step = global_step
            print('found %s, step from %d' % (model_ckpt_path, step))

        origin_step = step
        while True:
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # print(batch_data, batch_labels)
            session.run(
                [optimizer, loss],
                feed_dict={
                    x: batch_data,
                    y: batch_labels,
                    keep_prob: 0.5
                }
            )
            step += 1
            if step % 50 == 0:
                summary, test_accuracy = session.run(
                    [merged, accuracy],
                    feed_dict={
                        x: test_dataset,
                        y: test_labels,
                        keep_prob: 1.0
                    }
                )
                writer.add_summary(summary, step)
                print(("Step %4d, Test accuracy: %.4f" %
                       (step, test_accuracy)))
                if test_accuracy > 0.9955 or step - origin_step > 500:
                    if test_accuracy <= 0.9955:
                        print('you can re-format dataset and give a smaller alpha '
                              'to continue training')
                    else:
                        print('training done.')
                    save_model(step)
                    break

            else:
                summary, train_accuracy = session.run(
                    [merged, accuracy],
                    feed_dict={
                        x: batch_data,
                        y: batch_labels,
                        keep_prob: 1.0
                    }
                )
                writer.add_summary(summary, step)

            if step % 100 == 0:  # save the model every 100 step
                save_model(step)

        print("Test accuracy: %g" %
              session.run(
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

    kwargs = parser.parse_args().__dict__
    # print(kwargs)
    train(**kwargs)


if __name__ == '__main__':
    cli()
