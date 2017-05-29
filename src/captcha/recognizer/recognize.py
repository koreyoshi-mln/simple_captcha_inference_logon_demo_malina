#!/usr/bin/env python
# coding:utf-8

from __future__ import print_function

import os
import sys

from _recognize_p import start_recognize_daemon

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_dir)
from common.common import NNType


def recognize(captcha_path_set, nn_type=NNType.cnn):
    p = start_recognize_daemon(nn_type=nn_type)

    result_set = []
    for captcha_path in captcha_path_set:
        p.send(captcha_path)
        try:
            p.stdin.flush()
        except OSError:
            cracked = True
        else:
            cracked = False

        if cracked:
            print(p.recv_err())
            raise OSError('the recognize daemon process cracked up :(')
        result = p.recv()
        result_set.append(result)
        # print("result:", result)

    p.stdin.write(b'$exit\n')

    p.kill()

    return result_set


def cli():
    import sys
    if sys.argv[1] == 'cnn':
        nn_type = NNType.cnn
    else:
        nn_type = NNType.rnn
    captcha_path_set = sys.argv[2:]
    # print(captcha_path_set)

    result_list = recognize(captcha_path_set, nn_type=nn_type)
    for result in result_list:
        print(result)


if __name__ == '__main__':
    cli()
