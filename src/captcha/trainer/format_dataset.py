#!/usr/bin/env python
# coding:utf-8
from __future__ import division
from __future__ import print_function

import io
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

trainer_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(trainer_dir)
sys.path.append(home_dir)
from common.common import IMAGE_SIZE
from dataset import DataSet


def load_dataset():
    dataset = []
    labelset = []
    label_map = {}

    base_dir = os.path.join(trainer_dir, "training_set")
    index = 0
    for label in os.listdir(base_dir):
        if label.upper() == "ERROR" or label == ".DS_Store":
            continue
        if label.startswith('_'):  # Windows case insensitive, exp:
            # use dirname `_A` and `a` instead of `A` and `a`
            label = label[1:]

        print("loading:", label, "index:", index)
        try:
            image_files = os.listdir(os.path.join(base_dir, label))
            for image_file in image_files:
                image_path = os.path.join(base_dir, label, image_file)
                im = Image.open(image_path).convert('L')
                dataset.append(np.asarray(im, dtype=np.float32))
                # im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # dataset.append(im)
                labelset.append(index)
            label_map[index] = label
            index += 1
        except:
            raise

    return np.array(dataset), np.array(labelset), label_map


def _format_dataset(dataset, labels, image_size, num_labels): # one hot pattern
    dataset = dataset.reshape((-1, image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


DEFAULT_FORMATTED_DATATSET_DIR = trainer_dir


def format_dataset(formatted_dataset_dir=DEFAULT_FORMATTED_DATATSET_DIR,
                   log_file=io.StringIO()):

    dataset, labels, label_map = load_dataset()
    print("randomizing the dataset...", file=log_file)

    print("train_test_split the dataset...", file=log_file)
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels)

    print("reformating the dataset...", file=log_file)
    train_data, train_labels = _format_dataset(train_data, train_labels, IMAGE_SIZE, len(label_map))
    test_data, test_labels = _format_dataset(test_data, test_labels, IMAGE_SIZE, len(label_map))
    print("train_data:", train_data.shape, file=log_file)
    print("train_labels:", train_labels.shape, file=log_file)
    print("test_data:", test_data.shape, file=log_file)
    print("test_labels:", test_labels.shape, file=log_file)

    print("pickling the dataset...", file=log_file)

    formatted_train_dataset_path = os.path.join(formatted_dataset_dir, 'train_dataset.pickle')
    train_dataset = DataSet(train_data, train_labels, label_map)
    with open(formatted_train_dataset_path, 'wb') as f:
        pickle.dump(train_dataset, f, protocol=2)  # for compatible with python27

    formatted_test_dataset_path = os.path.join(formatted_dataset_dir, 'test_dataset.pickle')
    test_dataset = DataSet(test_data, test_labels, label_map)
    with open(formatted_test_dataset_path, 'wb') as f:
        pickle.dump(test_dataset, f, protocol=2)

    label_map_path = os.path.join(formatted_dataset_dir, 'label_map.pickle')
    with open(label_map_path, 'wb') as f2:
        pickle.dump(label_map, f2, protocol=2)

    print("dataset has saved at %s" % formatted_dataset_dir, file=log_file)
    print("load_model has finished", file=log_file)


def cli():
    import sys
    if len(sys.argv) > 1:
        formatted_dataset_dir = sys.argv[1]
        format_dataset(formatted_dataset_dir, sys.stdout)
    else:
        format_dataset(log_file=sys.stdout)


if __name__ == '__main__':
    cli()
