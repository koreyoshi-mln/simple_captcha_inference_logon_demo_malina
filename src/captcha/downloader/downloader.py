# -*- coding:utf-8 -*-
# !/usr/bin/env python3

"""

"""
import os
import sys
import time

from selenium import webdriver

index_url = 'http://ecard.xidian.edu.cn/index.jsp'
downloader_dir = os.path.dirname(os.path.abspath(__file__))
captchas_dir = os.path.join(downloader_dir, 'captchas')


def load_logon_page(driver):
    driver.get(index_url)

    for elem_a in driver.find_elements_by_tag_name('a'):
        if '登录' in elem_a.text:
            elem_a.click()
            break


def fetch_captcha_batch(driver, n=500, outdir=captchas_dir):
    load_logon_page(driver)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for i in range(n):
        time.sleep(0.5)
        elem_captcha_img = driver.find_element_by_id('myTab1_Content0').find_elements_by_tag_name('img')[0]
        captcha_path = os.path.join(outdir, 'captcha_' + str(i) + '.png')
        elem_captcha_img.screenshot(captcha_path)
        print(captcha_path)
        elem_captcha_img.click()


def cli():
    if len(sys.argv) > 1:
        n = sys.argv[1]
    else:
        n = 500

    driver = webdriver.Firefox()
    fetch_captcha_batch(driver, n=n)


if __name__ == '__main__':
    cli()
