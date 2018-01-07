#!user/bin/env python3
# -*- coding: utf-8 -*

'''
filename: download.py
'''

__author__ = 'JohnHuiWB'

from urllib import request, parse
from http import cookiejar
import time
import random
import socket
from datetime import datetime



# Throttle downloading by sleeping between requests to same domain
class Throttle:
    def __init__(self, delay):
        # amount of delay between downloads for each domain
        self._delay = delay
        # timestamp of when a domain last accessed
        self.domains = {}


    # delay if have accessed this domain recently
    def wait(self, url):
        domain = parse.urlparse(url).netloc
        last_accessed = self.domains.get(domain)

        if self._delay > 0 and last_accessed is not None:
            sleep_secs = self._delay - (datetime.now() - last_accessed).seconds
            if sleep_secs > 0:
                time.sleep(sleep_secs)
        self.domains[domain] = datetime.now()



class Downloader:
    def __init__(self, headers = None, delay = 3, proxies = None, num_retries = 2, cache = None, timeout = 60, opener = None, charset = 'utf-8', cookie = False):

        socket.setdefaulttimeout(timeout)

        self._throttle = Throttle(delay)
        self._headers = headers
        self._proxies = proxies
        self._num_retries = num_retries
        self._cache = cache
        self._opener = opener
        self._charset = charset
        self._cookie = cookie
        self._code = None


    def __call__(self, url, data = None):
        result = None
        if self._cache:
            try:
                result = self._cache[url]
            except KeyError:
                # url is not available in cache
                pass
            else:
                if self._num_retries > 0 and 500 <= result['code'] < 600:
                    # server error so ignore result from cache and re-download
                    result = None
        if result is None:
            # result was not loaded from cache, so still need to download
            self._throttle.wait(url)
            proxy = random.choice(self._proxies) if self._proxies else None
            headers = self._headers if self._headers else None
            result = self._download(url, proxy, self._num_retries, data, headers)
            if self._cache:
                # save result to cache
                self._cache[url] = result
        return result['html']


    def _download(self, url, proxy, num_retries, data, headers):
        print('='*120)
        print('Downloading:', url)
        print('proxy:', proxy)
        print('headers:', headers)
        print('='*120)

        req = request.Request(url, data, headers or {})

        if not self._opener:
            if proxy:
                proxy_handler = request.ProxyHandler(proxy)
                proxy_auth_handler = request.ProxyBasicAuthHandler()
                opener = request.build_opener(proxy_handler, proxy_auth_handler)
            else:
                opener = request.build_opener()
            if self._cookie:
                cj = cookiejar.CookieJar()
                handler = request.HTTPCookieProcessor(cj)
                opener.add_handler(handler)
        else:
            opener = self._opener

        try:
            response = opener.open(req)
            html = response.read()
            # decode the webpage
            html = html.decode(self._charset, 'ignore')

            self._code = response.code
        except request.URLError as e:
            print('Download error:', str(e))
            html = ''

            if hasattr(e, 'code') and 500 <= e.code < 600:
                # retry 5XX HTTP errors
                if num_retries > 0:
                    return self._download(url, proxy, num_retries - 1, data, headers)
            elif hasattr(e, 'code'):
                print('The server couldn\'t fulfill the request.')
                print('Error code:', e.code)
            elif hasattr(e, 'reason'):
                print('We failed to reach a server.')
                print('Error reason:', e.reason)

        return {'html': html, 'code': self._code}
