#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import print_function

import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

trainer_dir = os.path.dirname(os.path.abspath(__file__))


def _check_dataset(dataset, labels, label_map, index):
    data = np.uint8(dataset[index]).reshape((32, 32))
    i = np.argwhere(labels[index] == 1)[0][0]
    import matplotlib.pyplot as plt  # im.show may not be implemented
    #  in opencv-python on Tk GUI (such as Linux)
    import pylab
    plt.ion()
    plt.imshow(data)
    pylab.waitforbuttonpress(timeout=5)
    print("label:", label_map[i])


def check_dataset(path):
    with open(path, 'rb') as f:
        import sys
        if sys.version_info.major == 3:
            dataset = pickle.load(f, encoding='latin1')
        else:
            dataset = pickle.load(f)

            # check if the image is corresponding to it's label
    _check_dataset(dataset.images, dataset.labels, dataset.label_map, 0)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        formatted_dataset_dir = sys.argv[1]
    else:
        formatted_dataset_dir = trainer_dir

    train_dataset_path = os.path.join(formatted_dataset_dir, "train_dataset.pickle")
    test_dataset_path = os.path.join(formatted_dataset_dir, "test_dataset.pickle")

    # check if the image is corresponding to it's label
    check_dataset(train_dataset_path)
    check_dataset(test_dataset_path)
