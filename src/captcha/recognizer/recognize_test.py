#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

from recognize import recognize
home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_dir)
from common.common import NNType


recognize_dir = os.path.dirname(os.path.abspath(__file__))


def cli():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-p', '--path', default='test_set',
                        help='test set path')
    parser.add_argument('-nn', '--nn-type',  default='cnn', choices=['cnn', 'rnn'],
                        help='select neural network model type')

    kwargs = parser.parse_args().__dict__
    if kwargs['nn_type'] == 'cnn':
        nn_type = NNType.cnn
    elif kwargs['nn_type'] == 'rnn':
        nn_type = NNType.rnn

    path = kwargs['path']

    captcha_list = []
    for fn in os.listdir(path):
        # name, ext = os.path.splitext(fn)
        captcha_list.append(os.path.join(path, fn))

    result_list = recognize(captcha_list, nn_type=nn_type)
    correct = 0
    for path, result in zip(captcha_list, result_list):
        label = os.path.splitext(os.path.basename(path))[0][:4]
        print('%04s %04s ' % (label, result), end='')

        if label.replace('I', 'l').lower() == result.replace('I', 'l').lower():
            print(True)
            correct += 1
        else:
            print(False)

    print('accuracy: %f' % (correct / len(captcha_list)))


if __name__ == '__main__':
    cli()
