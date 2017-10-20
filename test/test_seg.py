#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170910
@author: JohnHuiWB
'''

import sys
sys.path.append("../")
from SynXNLP.seg.seg import *


S = Seg()
with open('test_seg_1.txt', 'r', encoding='utf8') as fp:
    text = fp.read()
S(text)
with open('test_seg_2.txt', 'r', encoding='utf8') as fp:
    text = fp.read()
S(text)