#!user/bin/env python3
# -*- coding: utf-8 -*

'''
Created on 20180107
@author: JohnHuiWB
'''

from SynXNLP.crawler.download import Downloader
import lxml.html


class Webcrawler(object):
    def __init__(self, t_charset = 'utf-8'):
        self.D = Downloader(charset = t_charset)


    def __call__(self, url:str):
        html = self.D(url)
        tree = lxml.html.fromstring(html)
        elements = tree.cssselect('body')
        return elements[0].text_content()
