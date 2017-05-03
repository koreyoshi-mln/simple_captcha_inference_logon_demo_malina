# -*- coding:utf-8 -*-
#!/usr/bin/env python3

"""logon demo
Usage:
  logon <username> <password> [--asyn-time=<asyn-time>]
  
Options
  --asyn-time=<asyn-time>  Firefox asyn time (s) [default: 0.5]
"""
from docopt import docopt

from logon import selenium_logon
def cli():
    arguments = docopt(__doc__)

    username = arguments['<username>']
    passwd = arguments['<password>']
    asyn_time = float(arguments['--asyn-time'])
    selenium_logon(username, passwd, asyn_time=asyn_time)

if __name__ == '__main__':
    cli()