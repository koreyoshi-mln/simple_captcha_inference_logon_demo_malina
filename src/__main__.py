# -*- coding:utf-8 -*-
#!/usr/bin/env python3

"""

"""
import os
import sys
from subprocess import Popen, PIPE
CLIENT_FILE = 'captcha_recognise_gui_controller.py'
def main():
    popen = None
    def start_gui(*args):
        p = Popen([sys.executable, CLIENT_FILE, *args], stdin=PIPE, stdout=PIPE)
        #print('client pid ',launcher.popen.pid)
        return p

    while True:
        if os.path.exists('.url'):
            with open('.url') as fr:
                url=fr.readline()
            print(url)
            popen = start_gui(url)
        else:
            popen = start_gui()

        popen.wait()

if __name__ == '__main__':
   main()
