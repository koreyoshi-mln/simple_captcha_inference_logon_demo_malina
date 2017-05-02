# -*- coding:utf-8 -*-
#!/usr/bin/env python3

"""

"""
import os

from PIL import Image
import requests

User_Agent = ('Mozilla/5.0(Windows NT 6.3)'
              'AppleWebKit/537.36 (KHTML,like Gecko)'
              'Chrome/39.0.2171.95'
              'Safari/537.36')

Accept = ('text/html,application/xhtml+xml,'
                   'application/xml;'
                   'q=0.9,image/webp,*/*;q=0.8')

headers={'User-Agent':User_Agent,
         'Accept':Accept,
         'Connection':'keep-alive',}

USERNAME = 'username'
PASSWORD = 'password'
CAPTCHA = 'captcha'

class NotValidPathStr(BaseException):pass
def get_image(s:str, captcha_name='captcha', session:requests.Session=None,
              savedir=os.path.dirname(os.path.abspath(__file__))):

    from urllib.request import urlretrieve

    import re
    url_net = "(https?|ftp)://[a-zA-Z0-9+&@#/%?=~_|$!:,.;]*[a-zA-Z0-9+&@#/%=~_|$]"
    pattern_url_net = url_net
    if re.match(pattern_url_net, s) is not None:
        captcha_path = os.path.join(savedir, captcha_name)
        if session is None:
            if os.path.exists(captcha_name):
                os.remove(captcha_name)
            urlretrieve(s, filename=captcha_path)
        else:
            r=session.get(s)

            with open(captcha_path, 'wb') as imgFile:
                imgFile.write(r.content)

        with Image.open(captcha_path) as imgObj:
            newfilepath = captcha_path + '.'+imgObj.format.lower()

        if os.path.exists(newfilepath):
            os.remove(newfilepath)

        os.rename(captcha_path, newfilepath)
        with Image.open(newfilepath) as img:
            imgObj = img.copy()

        #imgObj.show()
        return imgObj, newfilepath
    elif os.path.isfile(s):
        with Image.open(s) as img:
            imgObj = img.copy()

        return imgObj, s
    else:
        raise NotValidPathStr(s)

class KwargsError(Exception):
    pass

def payment_xidian_edu_cn_captchaurl(session:None, **kwargs):
    if session is None:
        session = requests.session()

    index_url = 'http://payment.xidian.edu.cn/fontuserLogin'
    captcha_url = 'http://payment.xidian.edu.cn/authImage'

    resp = session.get(index_url, headers=headers)
    return captcha_url, resp.text, session

def payment_xidian_edu_cn_logon(session:None, **kwargs):
    if session is None:
        session = requests.session()

    try:
        nickName = kwargs[USERNAME]
        password = kwargs[PASSWORD]
        checkCode = kwargs[CAPTCHA]
    except KeyError:
        kwargs_ok = False
    else:
        kwargs_ok = True

    if not kwargs_ok:
        raise KwargsError(kwargs)


    data = {
        'nickName':nickName,
        'password':password,
        'checkCode':checkCode,
    }
    post_path = 'http://payment.xidian.edu.cn/fontuserLogin'
    resp =session.post(post_path, data=data, headers=headers)

    return resp.text