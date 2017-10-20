#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170910
@author: JohnHuiWB
'''

import sys
sys.path.append("../")
from SynXNLP.feature.feature_extraction import *

E = Extracter()
train_vec = E.get_train_vec()
print(train_vec['_train_vec'])