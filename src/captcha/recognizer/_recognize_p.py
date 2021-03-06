#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function

import io
import os
import sys
import traceback
from subprocess import Popen, PIPE

try:
    import cpickle as pickle
except ImportError:
    import pickle

import tensorflow as tf
import numpy as np
import cv2

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_dir)
from common.common import load_label_map, find_model_ckpt, IMAGE_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NNType
import common.load_model_nn as load_nn
from spliter.spliter import split_letters

image_size = IMAGE_SIZE
if sys.version_info.major == 2:
    input = raw_input


def _fetch_stream(print_func, *args, **other_kwargs):
    buffer = io.StringIO()
    other_kwargs.pop('file', None)
    print_func(*args, file=buffer, **other_kwargs)
    content = buffer.getvalue()
    return content


def recognize_char_p(nn_type=NNType.cnn):
    label_map = load_label_map()  # 加载label值对应的label
                                  # 比如0->0, 10->`a`
    # 加载神经网络模型
    if nn_type == NNType.cnn:
        model = load_nn.load_model_cnn()
    else:
        model = load_nn.load_model_rnn()

    x = model['x']
    keep_prob = model['keep_prob']
    saver = model['saver']
    prediction = model['prediction']
    graph = model['graph']
    model_ckpt_path, _, ok = find_model_ckpt(nn_type=nn_type)  # 寻找断点(checkpoint)路径
    print('All was well', file=sys.stderr)
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()  # 各模型变量初始化
        saver.restore(session, model_ckpt_path)  # 从断点恢复经过训练以后的神经网络各模型的值

        while True:
            sys.stdout.flush()
            captcha_path = input().strip()  # 从当前进程的标准输入中读取一行，作为验证码存储路径
            if captcha_path == '$exit':  # for close session
                break

            try:
                # 通过OpenCV2.imread方法读取验证码的灰度图，返回一个像素矩阵
                # 然后通过numpy.reshape的方法将矩阵变形为一维的特征向量
                im = np.reshape(cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE), IMAGE_SIZE)
            except Exception as ex:
                # 如果发生异常，则向所在进程的标准输出流，写一个换行符，即返回空串
                sys.stdout.write('\n')
                # 将错误信息从所在进程的标准错误流里抓出来，放在字符串err_msg里
                err_msg = _fetch_stream(traceback.print_stack)
                err_msg = err_msg.replace('\n', '@@@')
                # 将处理后的err_msg字符串写到标准错误流里
                print('@@@'.join([str(ex), err_msg]), file=sys.stderr)
            else:
                print('All was well', file=sys.stderr)  # for recv_err
                # 根据学习的结果，对传入的特征向量进行预测，得到label值
                label = prediction.eval(feed_dict={x: [im], keep_prob: 1.0}, session=session)[0]
                # 通过label_map[label] 得到label数值对应的数字或者英文字符
                # 将字符写入所在进程的标准输出流
                sys.stdout.write(label_map[label])
                sys.stdout.write('\n')


def recognize_p(nn_type=NNType.cnn):
    """ 
    captcha_path
    $exit to exit
    """
    # print("recognize_p")

    label_map = load_label_map()
    if nn_type == NNType.cnn:
        model = load_nn.load_model_cnn()
    else:
        model = load_nn.load_model_rnn()

    x = model['x']
    keep_prob = model['keep_prob']
    saver = model['saver']
    prediction = model['prediction']

    model_ckpt_path, _, ok = find_model_ckpt(nn_type=nn_type)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        saver.restore(session, model_ckpt_path)

        while True:
            sys.stdout.flush()
            captcha_path = input().strip()
            # print("_recoginze", captcha_path)
            if captcha_path == '$exit':  # for close session
                break

            try:
                # 将完整的验证码图片进行处理，分割成标准的训练样本式的单个字符的列表
                # 然后再将每个字符处理成特征向量
                formatted_letters = split_letters(captcha_path)
                if nn_type == NNType.cnn:
                    formatted_letters = [letter.reshape(image_size) for letter in formatted_letters]
            except Exception as ex:
                sys.stdout.write('\n')
                err_msg = _fetch_stream(traceback.print_stack)
                err_msg = err_msg.replace('\n', '@@@')
                print('@@@'.join([str(ex), err_msg]), file=sys.stderr)

            else:
                print('All was well', file=sys.stderr)  # for recv_err

                result = []
                for letter in formatted_letters:
                    label = prediction.eval(feed_dict={x: [letter], keep_prob: 1.0}, session=session)[0]
                    # 识别的单个字符按顺序组成整体的对验证码的识别结果
                    result.append(label_map[label])
                    sys.stdout.write(label_map[label])

                sys.stdout.write('\n')


# start a recognize daemon process
# for interactive in IPython
# p.send('test.gif')
# p.recv()
# p.close()


def send(self, msg):
    """Interactive Tools
    self: subprocess.Popen
    p.send('abc.png')
    send(p, 'abc.png')
    
    _read_time: in case of block forever for no SIGALARM on Windows 555
    """
    if sys.version_info.major == 2:
        Str = unicode
    else:
        Str = str

    if isinstance(msg, Str):
        msg = msg.encode('utf8')

    try:
        self.stdin.write(msg + b'\n')
        self.stdin.flush()
    except OSError:
        raise IOError('this process halted')

    _read_time = getattr(self, '_read_time', 0)
    if _read_time > 100:
        raise BufferError('Warning: may no have enough space in buffer')

    setattr(self, '_read_time', _read_time + 1)

    _read_err_time = getattr(self, '_read_err_time', 0)
    setattr(self, '_read_err_time', _read_err_time + 1)


def recv(self, readall=False):
    """return str/unicode"""

    _read_time = getattr(self, '_read_time', 0)
    if _read_time == 0:
        raise IOError('you should send a value before recv')

    if readall:
        msg_list = []
        for i in range(_read_time):
            msg_list.append(self.stdout.readline().strip().decode())
        msg = ''.join(msg_list)
        _read_time = 0
    else:
        msg = self.stdout.readline().strip().decode()
        _read_time -= 1

    setattr(self, '_read_time', _read_time)
    return msg


def recv_err(self):
    _read_err_time = getattr(self, '_read_err_time', 0)
    if _read_err_time == 0:
        raise IOError('you should send a value before recv')

    err_msg = self.stderr.readline().strip().decode()
    err_msg = err_msg.replace('@@@', '\n')

    setattr(self, '_read_err_time', _read_err_time - 1)
    return err_msg


def close(self):
    self.stdin.write(b'$exit\n')
    self.kill()


def enhance_popen(p):
    from types import MethodType

    p.send = MethodType(send, p)
    p.recv = MethodType(recv, p)
    p.close = MethodType(close, p)
    p.recv_err = MethodType(recv_err, p)

    return p


__p_recognize = None  # private var!!!


def _close_recognize_process():
    if __p_recognize is not None:
        __p_recognize.send('$exit')
        __p_recognize.kill()


def start_recognize_char_daemon(nn_type=NNType.cnn):  # singleton include recognize_char because of saver.restore
    global __p_recognize
    if __p_recognize is not None and __p_recognize.poll() is None:
        raise OSError('the checkpoint is used by another reconize process')
    else:
        model_ckpt_path, _ = find_model_ckpt(nn_type=nn_type)[:2]
        print('load check-point %s' % model_ckpt_path)
        if nn_type == NNType.cnn:
            nn_type_s = 'cnn'
        else:
            nn_type_s = 'rnn'
        p = Popen([sys.executable, __file__, 'recognize_char', '-nn', nn_type_s],
                  bufsize=102400,
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # p.stdin.encoding = 'utf8'  # so we get `str` instead of `bytes` in p
        p = enhance_popen(p)
        __p_recognize = p
        return p


def start_recognize_daemon(nn_type=NNType.cnn):  # singleton
    global __p_recognize
    if __p_recognize is not None and __p_recognize.poll() is None:
        raise OSError('the checkpoint is used by another reconize process')
    else:
        model_ckpt_path, _ = find_model_ckpt(nn_type=nn_type)[:2]
        print('load check-point %s' % model_ckpt_path)
        if nn_type == NNType.cnn:
            nn_type_s = 'cnn'
        else:
            nn_type_s = 'rnn'

        p = Popen([sys.executable, __file__, 'recognize', '-nn', nn_type_s],
                  bufsize=102400,
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # p.stdin.encoding = 'utf8'  # so we get `str` instead of `bytes` in p
        p = enhance_popen(p)
        __p_recognize = p
        return p


def cli():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('recognize_type', choices=['recognize_char', 'recognize'],
                        help='recognize type')

    parser.add_argument('-nn', '--nn-type', default='cnn', choices=['cnn', 'rnn'],
                        help='select neural network model type')
    kwargs = parser.parse_args().__dict__
    if kwargs['nn_type'] == 'cnn':
        nn_type = NNType.cnn
    elif kwargs['nn_type'] == 'rnn':
        nn_type = NNType.rnn
    # print(sys.argv)

    if kwargs['recognize_type'] == 'recognize_char':
        recognize_char_p(nn_type=nn_type)
    else:
        recognize_p(nn_type=nn_type)


if __name__ == '__main__':
    cli()
