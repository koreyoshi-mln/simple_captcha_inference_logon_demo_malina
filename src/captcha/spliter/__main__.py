#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function

import os

from spliter import split_test

spliter_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(spliter_dir)


def cli():
    captchas_path = os.path.join(home_dir, 'downloader', 'captchas')
    dataset_path = os.path.join(spliter_dir, "dataset")

    split_test(input_dir=captchas_path, out_dir=dataset_path)


if __name__ == '__main__':
    cli()
