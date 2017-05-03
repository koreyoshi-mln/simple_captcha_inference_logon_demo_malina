# -*- coding:utf-8 -*-
#!/usr/bin/env python3

"""

"""
import time

from selenium import webdriver

from captcha.recognizer._recognize_p import start_recognize_daemon

login_url = 'http://ecard.xidian.edu.cn/cardUserManager.do?method=checkLogin'
index_url = 'http://ecard.xidian.edu.cn/index.jsp'

def load_logon_page(driver):
    driver.get(index_url)

    for elem_a in driver.find_elements_by_tag_name('a'):
        if '登录' in elem_a.text:
            elem_a.click()
            break

def is_loggedin(driver):
    for elem_a in driver.find_elements_by_tag_name('a'):
        if '登录' in elem_a.text:
            return False

    return True

def logout(driver):
    elem_logout=None
    for elem in driver.find_elements_by_tag_name('a'):
        if elem.text.strip() == '退出':
            elem_logout=elem
            break

    elem_logout.click()

def refresh(driver):
    driver.back()
    driver.refresh()

def selenium_logon(card_id, card_passwd, asyn_time = 0.5):
    driver = webdriver.Firefox()
    p = start_recognize_daemon()
    while True:
        load_logon_page(driver)
        time.sleep(asyn_time)

        while True:
            try:
                elem_cardid = driver.find_elements_by_id('code')[0]
            except:
                time.sleep(asyn_time)
            else:
                break
        elem_cardid.clear()
        elem_cardid.send_keys(card_id)

        elem_pwd = driver.find_elements_by_id('pwd')[0]
        elem_pwd.clear()
        elem_pwd.send_keys(card_passwd)

        elem_captcha_img = driver.find_element_by_id('myTab1_Content0').find_elements_by_tag_name('img')[0]
        elem_captcha_img.screenshot('captcha.png')

        elem_captcha = driver.find_elements_by_id('cardCheckCode')[0]
        elem_captcha.clear()
        #TODO inference here
        p.send('captcha.png')
        result = p.recv()
        if result == '': # recognize failed
            print(p.recv_err())
        elem_captcha.send_keys(result)
        time.sleep(2) #wait to look

        elem_login=driver.find_element_by_id('myTab1_Content0').find_elements_by_class_name('button_denglu')[0]
        elem_login.click()

        time.sleep(2)

        try:
            if is_loggedin(driver):
                p.close()
                return #TODO after login
            else: #captcha error, continue
                refresh(driver)
        except:
            refresh(driver)




















